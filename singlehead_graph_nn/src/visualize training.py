import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("../docs/training_results/GSGAT_5class_0509.csv")

# Create a figure with 3 subplots
fig, axs = plt.subplots(3, 1, figsize=(20, 25), constrained_layout=True)

# Plot loss in the first subplot
axs[0].plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue')
axs[0].plot(df['epoch'], df['val_loss'], label='Validation Loss', color='red')
axs[0].set_title('Loss')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Loss')
axs[0].legend()

# Plot accuracy in the second subplot
axs[1].plot(df['epoch'], df['train_acc'], label='Train Accuracy', color='blue')
axs[1].plot(df['epoch'], df['val_acc'], label='Validation Accuracy', color='red')
axs[1].set_title('Accuracy')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Accuracy')
axs[1].legend()

# Plot F1 score in the third subplot
axs[2].plot(df['epoch'], df['train_f1'], label='Train F1 Score', color='blue')
axs[2].plot(df['epoch'], df['val_f1'], label='Validation F1 Score', color='red')
axs[2].set_title('F1 Score')
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('F1 Score')
axs[2].legend()

# Adjust layout
# plt.tight_layout()
plt.savefig("../docs/training_results/graph_sage_0508.png")
# Show the figure
plt.show()

