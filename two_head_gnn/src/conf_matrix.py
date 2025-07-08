import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import config

# Load the results JSON
with open('../docs/TwoHead_Evaluation_results/TwoHeadGAT_small_2025_7_4/predictions_labels_1452.json', 'r') as f:
    results = json.load(f)

# Extract predictions and ground truths
wire_pred = results['predicted_wire']
wire_true = results['wire_truth']
action_pred = results['predicted_action']
action_true = results['action_truth']

# Function to print metrics and plot confusion matrix
def evaluate_and_plot(true, pred, label):
    print(f"\n=== {label.upper()} Evaluation ===")
    print(classification_report(true, pred, digits=3))

    cm = confusion_matrix(true, pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                xticklabels=sorted(set(true)),
                yticklabels=sorted(set(true)))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{label.capitalize()} Confusion Matrix')
    plt.tight_layout()
    plt.show()

# Evaluate each category
evaluate_and_plot(wire_true, wire_pred, 'wire')
evaluate_and_plot(action_true, action_pred, 'action')