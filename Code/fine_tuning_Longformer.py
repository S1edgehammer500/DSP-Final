from datasets import DatasetDict, Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, Trainer
from torch.optim import AdamW
import evaluate
import numpy as np
import torch.nn.functional as t
import torch
import pandas as pd
from collections import Counter
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import spacy
import os
import re

# add every script to a list of dictionaries
def buildingDataset():
  data_path = f"DSP-Final/movie_scripts"
  scripts = []

  for filename in os.listdir(data_path):
      if filename.endswith(".txt"):
          with open(os.path.join(data_path, filename), encoding="utf-8", errors="ignore") as f:
              text = f.read()
              scripts.append({"text": text})
  return scripts

scripts = buildingDataset()

# overwriting the Trainer class
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # set the loss function to include weights
        loss_fct = CrossEntropyLoss(weight=class_weights.to(model.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

    # create an optimizer with AdamW for learning rate decay
    def create_optimizer(self):
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=initial_lr)
        return self.optimizer

    def training_step(self, *args, **kwargs):
      output = super().training_step(*args, **kwargs)
      # empty cache
      torch.cuda.empty_cache()
      return output

    def evaluate(self, *args, **kwargs):
        output = super().evaluate(*args, **kwargs)
        # empty cache
        torch.cuda.empty_cache()
        return output


print(f"Original dataset size {len(scripts)}")
rating_map = {'U': 0, 'PG': 1, '12': 2, '12A': 3, '15': 4, '18': 5}
nlp = spacy.load("en_core_web_sm")
# disable uneseccary pipelines
nlp.disable_pipes("tagger", "ner", "lemmatizer", "attribute_ruler")

def clean_script(script):
    lines = [line.strip() for line in script["text"].splitlines() if line.strip()]

    # first non-empty line is assumed to be the age rating
    age_rating_line = lines[0]
    label = rating_map.get(age_rating_line.split(":")[1].strip())

    # remove first non-empty line (Age Rating)
    raw = "\n".join(lines[1:])

    return {"raw": raw, "label": label}

def process_all_scripts(scripts):
    # clean scripts and prepare text-label pairs
    cleaned_scripts = [clean_script(script) for script in scripts]

    # extract raw texts and labels
    raw_texts = [s["raw"] for s in cleaned_scripts]
    age_ratings = [s["label"] for s in cleaned_scripts]
    cleaned_texts = []
    for script in raw_texts:
      # clean the scripts into the desired format
      subbed_text = re.sub(r"(?<!\n)\n(?!\n)", ' ', script)
      cleaned_texts.append(subbed_text)

    # run the pipeline on the cleaned scripts
    docs = list(nlp.pipe(cleaned_texts, batch_size=64))
    processed_scripts = []
    for doc, label in zip(docs, age_ratings):
        # combine sentences back into one bit of text
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        cleaned_script = "\n".join(sentences)
        processed_scripts.append({"text": cleaned_script, "label": label})
    return processed_scripts

# split the script into multiple chunks
def chunk_script(text, tokenizer, max_length=4096, stride=512):
    tokens = tokenizer(text, return_attention_mask=False, return_token_type_ids=False, truncation=False)["input_ids"]
    chunks = []
    for i in range(0, len(tokens), max_length - stride):
        chunk = tokens[i:i+max_length]
        if len(chunk) < max_length:
            # pad the chunk if 4096 tokens isn't met
            chunk += [tokenizer.pad_token_id] * (max_length - len(chunk))
        chunks.append(chunk)
    return chunks

# create a dataset from the dictionary of chunked scripts
def create_chunked_dataset(processed_scripts, tokenizer):
    chunked_texts = []
    chunked_labels = []

    for script in processed_scripts:
        chunks = chunk_script(script["text"], tokenizer)
        chunked_texts.extend(chunks)
        chunked_labels.extend([script["label"]] * len(chunks))

    return Dataset.from_dict({"input_ids": chunked_texts, "label": chunked_labels})


# apply the processing function to the full list of scripts
processed_scripts = process_all_scripts(scripts)

# set the Longformer model
model_path = "allenai/longformer-base-4096"

# tokenize the data
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = "right"
tokenizer.truncation_side = "right"


# chunk the processed scripts using the tokenizer
chunked_dataset = create_chunked_dataset(processed_scripts, tokenizer)

# split the dataset into train/test
dataset = chunked_dataset.train_test_split(test_size=0.2)

# extract scripts and labels from the training portion
scripts = dataset["train"]["input_ids"]
labels = dataset["train"]["label"]

# split the training set into train and validation using sklearn, with stratification
train_scripts, val_scripts, train_labels, val_labels = train_test_split(scripts, labels, test_size=0.1, stratify=labels, random_state=42)

# convert splits back into HuggingFace Datasets
train_dataset = Dataset.from_dict({"input_ids": train_scripts, "label": train_labels})
val_dataset = Dataset.from_dict({"input_ids": val_scripts, "label": val_labels})
test_dataset = dataset["test"]  # already a Hugging Face Dataset

# wrap in DatasetDict
dataset_dict = DatasetDict({"train": train_dataset, "validation": val_dataset, "test": test_dataset})



# set up the model with the path and added layers
id2label = {rating: value for value, rating in rating_map.items()}
label2id = rating_map
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=6, id2label=id2label, label2id=label2id)

# calculate the class weights
label_counts = Counter([example["label"] for example in train_dataset])
print("Label distribution:", label_counts)
total = sum(label_counts.values())
class_weights = [total / label_counts[i] for i in range(len(rating_map))]
class_weights = torch.tensor(class_weights).to(torch.float)
print("Class weights:", class_weights)

# model config setup
model.config.problem_type = "single_label_classification"
model.config.classifier_dropout = 0.1

# set the data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# load in AUC score
auc_score = evaluate.load("roc_auc", config_name="multiclass")

# calculate prediction accuracy between 0-1
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    probabilities = t.softmax(torch.tensor(predictions), dim=-1).numpy()
    predicted_classes = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predicted_classes)
    precision = precision_score(labels, predicted_classes, average='weighted', zero_division=0)
    recall = recall_score(labels, predicted_classes, average='weighted', zero_division=0)
    f1 = f1_score(labels, predicted_classes, average='weighted', zero_division=0)

    num_classes = len(id2label)
    print("Probabilities shape:", probabilities.shape)
    print("Labels shape:", labels.shape)
    print("Sample probs:", probabilities[0])
    print("Sample label:", labels[0])


    # fallback if AUC is 0.0
    try:
        auc = np.round(auc_score.compute(prediction_scores=probabilities, references=labels, average='macro', multi_class="ovr")['roc_auc'], 3)
    except ValueError:
        auc = 0.0  

    predicted_classes = np.argmax(probabilities, axis=1)
    print("Predicted class counts:", Counter(predicted_classes))
    print("True class counts:", Counter(labels))

    # display the classification report after each epoch
    print(classification_report(labels, predicted_classes, target_names=id2label.values()))

    return {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1": f1, "AUC": auc}


# set the learning rate, batch size, and number of epochs
batch_size = 8
num_epochs = 5
weight_decay = 0.01
initial_lr = 2e-5
layerwise_lr_decay = 0.95
no_decay = ["bias", "LayerNorm.weight"]

# extract all encoder layers (from bottom to top)
encoder_layers = model.longformer.encoder.layer
num_layers = len(encoder_layers)

# helper to collect parameters layer by layer
optimizer_grouped_parameters = []

# add embedding layer with lowest LR
optimizer_grouped_parameters.append({
    "params": [p for n, p in model.longformer.embeddings.named_parameters() if not any(nd in n for nd in no_decay)],
    "weight_decay": weight_decay,
    "lr": initial_lr * (layerwise_lr_decay ** num_layers)})
optimizer_grouped_parameters.append({
    "params": [p for n, p in model.longformer.embeddings.named_parameters() if any(nd in n for nd in no_decay)],
    "weight_decay": 0.0,
    "lr": initial_lr * (layerwise_lr_decay ** num_layers)})

# add encoder layers with decaying LRs
for i, layer in enumerate(encoder_layers):
    layer_lr = initial_lr * (layerwise_lr_decay ** (num_layers - i - 1))
    optimizer_grouped_parameters.append({
        "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
        "lr": layer_lr,
    })
    optimizer_grouped_parameters.append({
        "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
        "lr": layer_lr,
    })

# add the classifier layer
optimizer_grouped_parameters.append({
    "params": [p for n, p in model.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
    "weight_decay": weight_decay,
    "lr": initial_lr})

optimizer_grouped_parameters.append({
    "params": [p for n, p in model.classifier.named_parameters() if any(nd in n for nd in no_decay)],
    "weight_decay": 0.0,
    "lr": initial_lr})

# reduces risk of OOM Error
model.gradient_checkpointing_enable()

# define the training arguments
training_args = TrainingArguments(output_dir="age_rating_classifier", learning_rate=initial_lr, per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size, num_train_epochs=num_epochs, logging_strategy="epoch", eval_strategy="epoch", save_strategy="epoch", weight_decay=0.01, bf16=True, load_best_model_at_end=True)

# train the model using these arguments
trainer = WeightedTrainer(model=model, args=training_args, train_dataset=dataset_dict["train"], eval_dataset=dataset_dict["validation"], tokenizer=tokenizer, data_collator=data_collator, compute_metrics=compute_metrics)
trainer.train()
torch.cuda.empty_cache()

log_history = trainer.state.log_history
metrics = []

# append the metrics to an Excel sheet
for entry in log_history:
    if "eval_Accuracy" in entry:
        metrics.append({"epoch": entry["epoch"], "accuracy": entry["eval_Accuracy"], "precision": entry["eval_Precision"], "recall": entry["eval_Recall"], "f1": entry["eval_F1"], "auc": entry["eval_AUC"],})

df = pd.DataFrame(metrics)
df.to_excel("training_results.xlsx", index=False)

# find the best model
best_model_path = trainer.state.best_model_checkpoint
print(best_model_path)
model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
tokenizer = AutoTokenizer.from_pretrained(best_model_path)


def predict_on_test_data():
    # create dataloader from test data
    test_dataloader = DataLoader(test_dataset, batch_size=8)

    # run predictions
    trainer = Trainer(model=model)
    predictions = trainer.predict(test_dataset)

    predicted_classes = np.argmax(predictions.predictions, axis=1)
    test_labels = predictions.label_ids

    report = classification_report(test_labels, predicted_classes, target_names=id2label.values())
    print(f"\nTest Set Evaluation:{report}")

    # save predictions
    results_df = pd.DataFrame({"true_label": test_labels, "predicted_label": predicted_classes})
    results_df.to_excel("test_predictions.xlsx", index=False)

# run the model on test data
predict_on_test_data()