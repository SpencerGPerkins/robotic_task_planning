# plot_metrics.py

import matplotlib.pyplot as plt

def plot_metric_subplot(ax, x, y1, y2, title, ylabel, label1, label2):
    ax.plot(x, y1, label=label1)
    ax.plot(x, y2, label=label2)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)

def plot_wire_metrics(training_results, save_path):
    epochs = training_results["epoch"]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Wire Classification Metrics", fontsize=16)

    plot_metric_subplot(axs[0], epochs, training_results["wire_train_acc"], training_results["wire_val_acc"],
                        "Accuracy", "Accuracy", "Train", "Val")
    plot_metric_subplot(axs[1], epochs, training_results["wire_train_f1"], training_results["wire_val_f1"],
                        "F1 Score", "F1", "Train", "Val")
    plot_metric_subplot(axs[2], epochs, training_results["wire_train_loss"], training_results["wire_val_loss"],
                        "Loss", "Loss", "Train", "Val")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f"{save_path}_wire.png")
    plt.close(fig)

def plot_action_metrics(training_results, save_path):
    epochs = training_results["epoch"]
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Action Classification Metrics", fontsize=16)

    plot_metric_subplot(axs[0], epochs, training_results["action_train_acc"], training_results["action_val_acc"],
                        "Accuracy", "Accuracy", "Train", "Val")
    plot_metric_subplot(axs[1], epochs, training_results["action_train_f1"], training_results["action_val_f1"],
                        "F1 Score", "F1", "Train", "Val")
    plot_metric_subplot(axs[2], epochs, training_results["action_train_loss"], training_results["action_val_loss"],
                        "Loss", "Loss", "Train", "Val")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f"{save_path}_action.png")
    plt.close(fig)

def plot_total_loss(training_results, save_path):
    epochs = training_results["epoch"]
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(epochs, training_results["train_loss"], label="Train Loss", color='blue')
    ax.plot(epochs, training_results["val_loss"], label="Val Loss", color='orange')
    ax.set_title("Total Training and Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    fig.savefig(f"{save_path}_total_loss.png")
    plt.close(fig)

def generate_all_plots(training_results, save_path):
    plot_wire_metrics(training_results, save_path)
    plot_action_metrics(training_results, save_path)
    plot_total_loss(training_results, save_path)
