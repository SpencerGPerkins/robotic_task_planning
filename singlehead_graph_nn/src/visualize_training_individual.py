import pandas as pd
import matplotlib.pyplot as plt


import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("../docs/training_results/graph_sage_0324.csv")

# Loss plot
plt.figure()
plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue')
plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', color='red')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("../docs/training_results/graph_sage_0321loss.png")
plt.close()

# Accuracy plot
plt.figure()
plt.plot(df['epoch'], df['train_acc'], label='Train Accuracy', color='blue')
plt.plot(df['epoch'], df['val_acc'], label='Validation Accuracy', color='red')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("../docs/training_results/graph_sage_0321acc.png")
plt.close()

# F1 Score plot
plt.figure()
plt.plot(df['epoch'], df['train_f1'], label='Train F1', color='blue')
plt.plot(df['epoch'], df['val_f1'], label='Validation F1', color='red')
plt.title('F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.savefig("../docs/training_results/graph_sage_0321f1.png")
plt.close()

