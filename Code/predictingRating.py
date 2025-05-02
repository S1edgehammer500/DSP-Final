import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import spacy
import pandas as pd
import heapq
import re

# load model and tokenizer
best_model_path = "DSP-Final/age_rating_classifier/age_rating_classifier/checkpoint-2340"
model = AutoModelForSequenceClassification.from_pretrained(best_model_path)
tokenizer = AutoTokenizer.from_pretrained(best_model_path)

# enable GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# load spaCy
nlp = spacy.load("en_core_web_sm")
nlp.disable_pipes("ner", "lemmatizer", "tagger", "attribute_ruler")

def preprocess_script(script_text):
    cleaned_text = re.sub(r"(?<!\n)\n(?!\n)", ' ', script_text)
    doc = nlp(cleaned_text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return "\n".join(sentences)

def classify_script(script_text):
    cleaned_text = preprocess_script(script_text)

    # tokenize the full script
    tokens = tokenizer(cleaned_text, return_tensors="pt", truncation=False, padding=False)

    input_ids = tokens["input_ids"][0]  # shape: [seq_len]
    attention_mask = tokens["attention_mask"][0]

    max_length = 4096
    chunks = []
    masks = []

    # chunk into blocks of 4096 tokens
    for i in range(0, input_ids.size(0), max_length):
        chunk_ids = input_ids[i:i+max_length]
        chunk_mask = attention_mask[i:i+max_length]

        # pad if needed
        if chunk_ids.size(0) < max_length:
            pad_len = max_length - chunk_ids.size(0)
            chunk_ids = torch.cat([chunk_ids, torch.zeros(pad_len, dtype=torch.long)])
            chunk_mask = torch.cat([chunk_mask, torch.zeros(pad_len, dtype=torch.long)])

        chunks.append(chunk_ids.unsqueeze(0))
        masks.append(chunk_mask.unsqueeze(0))

    # stack all chunks and send to device
    input_ids_tensor = torch.cat(chunks).to(device)
    attention_mask_tensor = torch.cat(masks).to(device)

    all_logits = []

    model.eval()
    with torch.no_grad():
        for i in range(input_ids_tensor.size(0)):
            outputs = model(
                input_ids=input_ids_tensor[i].unsqueeze(0),
                attention_mask=attention_mask_tensor[i].unsqueeze(0)
            )
            all_logits.append(outputs.logits)

    # average logits over all chunks
    avg_logits = torch.mean(torch.stack(all_logits), dim=0)
    probs = torch.nn.functional.softmax(avg_logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()

    id2label = model.config.id2label
    return id2label[predicted_class], probs.cpu().numpy().flatten()

st.title("Movie Age Rating Predicter")
uploaded_file = st.file_uploader("Upload a script (.txt)", type=["txt"])

def highlight_top_blocks(script_text, model, tokenizer, target_label, top_k=20, batch_size=16):
    all_lines = [line for line in script_text.splitlines() if line.strip()]
    total = len(all_lines)
    
    # create blocks of 5 lines
    blocks = [
        "\n".join(all_lines[i:i+5])
        for i in range(0, total, 5)
        if len(all_lines[i:i+5]) > 0
    ]
    
    block_scores = []
    progress_bar = st.progress(0)

    # run the model on all line sets in batches
    model.eval()
    with torch.no_grad():
        batches = [blocks[i:i+batch_size] for i in range(0, len(blocks), batch_size)]
        for batch_idx, batch in enumerate(batches):
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=512
            )

            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            label_idx = model.config.label2id[target_label]

            for block_text, prob in zip(batch, probs):
                score = prob[label_idx].item()
                block_scores.append((block_text, score))

            progress_bar.progress(min((batch_idx + 1) / len(batches), 1.0))

    # top-K blocks
    top_blocks = heapq.nlargest(top_k, block_scores, key=lambda x: x[1])
    top_block_set = set(text for text, _ in top_blocks)

    # re-highlight full script
    highlighted_script = ""
    for i in range(0, total, 5):
        block = "\n".join(all_lines[i:i+5])
        safe_block = escape_markdown(block)
        if block in top_block_set:
            highlighted_script += f":orange[{safe_block}]\n\n"
        else:
            highlighted_script += safe_block + "\n\n"

    return highlighted_script




if uploaded_file:
    script_text = uploaded_file.read().decode("utf-8")

    lines = script_text.splitlines()

    # remove the first non-empty line if it starts with "UK Age Rating:"
    filtered_lines = []
    found_rating = False

    for line in lines:
        stripped = line.strip()
        if not found_rating and stripped.startswith("UK Age Rating:"):
            found_rating = True
            continue  # Skip this line
        filtered_lines.append(line)

    # join cleaned lines back
    script_text = "\n".join(filtered_lines)
    
    preview_placeholder = st.empty()
    preview_placeholder.text_area("Original Script Preview", script_text[:3000], height=300)

    with st.spinner("Analyzing script..."):
        predicted_label, probs = classify_script(script_text)
        
    rating_order = ["U", "PG", "12", "12A", "15", "18"]


    confidence_df = pd.DataFrame({
        "Age Rating": list(model.config.id2label.values()),
        "Confidence": probs
    })

    confidence_df["Age Rating"] = pd.Categorical(confidence_df["Age Rating"], categories=rating_order, ordered=True)

    # sort the DataFrame by the defined order
    confidence_df = confidence_df.sort_values("Age Rating")

    st.subheader(f"Predicted Age Rating: {predicted_label}")

    # show bar chart of confidence scores
    st.bar_chart(confidence_df.set_index("Age Rating"))

    def escape_markdown(text):
        # escape hyphen, asterisk, and numbered list markdown at the start of lines
        return re.sub(r'^(\s*[-*]|\s*\d+\.)', r'\\\1', text, flags=re.MULTILINE)

    with st.spinner("Highlighting most influential lines..."):
        highlighted_script = highlight_top_blocks(script_text, model, tokenizer, predicted_label, top_k=20)
    st.markdown("### Top Influential Lines (highlighted in orange)")
    st.markdown(highlighted_script, unsafe_allow_html=True)