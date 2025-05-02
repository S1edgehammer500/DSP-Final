import pandas as pd

# Load Excel file
df = pd.read_excel("DSP-Final/test_predictions.xlsx")  

# Initialize a dictionary to count incorrect predictions per label
label_map = {0: 'U', 1: 'PG', 2: '12', 3: '12A', 4: '15', 5: '18'}
incorrect_counts = {label: 0 for label in label_map.keys()}
correct_counts = {label: 0 for label in label_map.keys()}

# Loop through rows to count incorrect predictions
for index, row in df.iterrows():
    true = row['true_label']
    pred = row['predicted_label']
    if true != pred:
        incorrect_counts[true] += 1
    else:
        correct_counts[true] += 1

readable_counts = {label_map[k]: v for k, v in incorrect_counts.items()}
readbale_correct_counts = {label_map[k]:v for k, v in correct_counts.items()}

# Print the result
print("Incorrect prediction counts per rating:")
for label, count in readable_counts.items():
    print(f"{label}: {count}")

print(f"Correct predictions per rating:")
for label, count in readbale_correct_counts.items():
    print(f"{label}: {count}")
