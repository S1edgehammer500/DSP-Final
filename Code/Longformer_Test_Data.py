from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
import evaluate
import numpy as np
import pandas as pd
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from torch.utils.data import DataLoader
from fine_tuning_Longformer import test_dataset, id2label

best_model_path = "DSP-Final/age_rating_classifier/age_rating_classifier/checkpoint-2340"
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
    print(f"\nTest Set Evaluation:\n{report}")

    # save predictions
    results_df = pd.DataFrame({
        "true_label": test_labels,
        "predicted_label": predicted_classes
    })
    accuracy = accuracy_score(test_labels, predicted_classes)
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    results_df.to_excel("test_predictions.xlsx", index=False)

# run the model on test data
predict_on_test_data()