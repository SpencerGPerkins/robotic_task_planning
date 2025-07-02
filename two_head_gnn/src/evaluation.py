import torch
from torch_geometric.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from datetime import datetime
import os
import pandas as pd
import json

import config
from models import TwoHeadGATSmall, TwoHeadGAT
from data_process import load_dataset
from utils import load_checkpoint

month = datetime.now().month
day = datetime.now().day
year = datetime.now().year
hour = datetime.now().hour
minute = datetime.now().minute

def eval(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    wire_loss = 0
    action_loss = 0

    all_wire_preds, all_wire_labels = [], []
    all_action_preds, all_action_labels = [], []

    loss_iter_counter = 0
    with torch.no_grad():
        for data in loader:
            # Node mask
            wire_mask = data.wire_mask
            data.to(device)
            wire_logits, action_logits = model(
                data.x.float(), wire_mask, data.edge_index, data.edge_attr, data.batch
            )
            wire_label_local = torch.tensor(data.y_wire_local.squeeze()).to(device)
            action_label = torch.tensor(data.y_action).float().to(device)

            if action_label.ndim == 2: # Shape [B, 4]
                action_label = action_label.argmax(dim=1)
            else: # shape [4]
                action_label = action_label.unsqueeze(0).argmax(dim=1)

            # Loss weights
            wire_weight = 1.0
            action_weight = 2.0

            wire_loss = criterion(wire_logits, wire_label_local)
            wire_loss += wire_loss.item()
            action_loss = criterion(action_logits, action_label)
            action_loss += action_loss.item()

            loss = (wire_weight * wire_loss) + (action_weight * action_loss)
            total_loss += loss.item()
            if action_label.item() == 2:
                print(f"Lock sample number {loss_iter_counter}")
                print(f"Lock Action loss: {loss}")
                loss_iter_counter +=1

            # Predictions
            wire_pred_local = wire_logits.argmax().item()
            wire_label_local = wire_label_local.item()
            all_wire_preds.append(wire_pred_local)
            all_wire_labels.append(wire_label_local)
            all_action_preds.append(action_logits.argmax().item())
            all_action_labels.append(action_label.item())

    # Accuracy per head
    wire_acc = accuracy_score(all_wire_labels, all_wire_preds)
    action_acc = accuracy_score(all_action_labels, all_action_preds)

    # F1 Scores
    wire_f1 = f1_score(all_wire_labels, all_wire_preds, average="weighted")
    action_f1 = f1_score(all_action_labels, all_action_preds, average="weighted")

    # Loss

    averaged_loss = total_loss / len(loader)

    averaged_wire_loss = wire_loss / len(loader)
    averaged_action_loss = action_loss / len(loader)

    preds_labels = {
        "predicted_wire": all_wire_preds,
        "wire_truth": all_wire_labels,
        "predicted_action":all_action_preds,
        "action_truth":all_action_labels
    }

    results_metrics = {
        "aver_loss": [averaged_loss],
        "wire_acc": [wire_acc],
        "wire_f1": [wire_f1],
        "aver_wire_loss": [averaged_wire_loss.item()],
        "action_acc": [action_acc],
        "action_f1": [action_f1],
        "averaged_action_loss": [averaged_action_loss.item()]   
    }

    return results_metrics, preds_labels

def main():
    vision_data = config.EVAL_VISION_DATA_PATH
    llm_data = config.EVAL_LLM_DATA_PATH
    label_data = config.EVAL_LABEL_DATA_PATH
    num_samples = config.NUM_EVAL_SAMPLES

    print("\n\nLoading Data...\n\n")

    dataset = load_dataset(vision_data, llm_data, label_data, num_data_samples=num_samples, action_primitives=config.ACTION_PRIMS)

    eval_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = config.DEVICE
    
    if config.MODEL_SIZE == "small":
        model = TwoHeadGATSmall(in_dim=len(dataset[0].x[0]), edge_feat_dim=1, hidden_dim=config.HIDDEN_DIM, num_actions=config.NUM_ACTIONS).to(device)
        print("\n\nEvaluating Small Model...\n\n")
    elif config.MODEL_SIZE == "medium":
        model = TwoHeadGAT(in_dim=len(dataset[0].x[0]), edge_feat_dim=1, hidden_dim=config.HIDDEN_DIM, num_actions=config.NUM_ACTIONS).to(device)
        print("\n\nMedium Model Used...\n\n")

    if os.path.exists(config.CHECKPOINT_PATH):
        model = load_checkpoint(config.CHECKPOINT_PATH, model, torch.load)
    else:
        raise RuntimeError(f"\n\nCheckpoint not found at {config.CHECKPOINT_PATH}. NO MODEL TO EVALUATE.\n\n")  
    
    criterion = torch.nn.CrossEntropyLoss()
    metrics, predictions_labels = eval(model, eval_loader, criterion, device)
    metrics_df = pd.DataFrame.from_dict(metrics)
    results_path = f"{config.SAVE_EVAL_RESULTS_HEAD}_{year}_{month}_{day}/"
    os.makedirs(results_path, exist_ok=True)
    metrics_df.to_csv(f"{results_path}metrics_{hour}{minute}.csv")
    with open(f"{results_path}predictions_labels_{hour}{minute}.json", "w") as pred_labs:
        json.dump(predictions_labels, pred_labs)
    print(f"\nEvaluation on {config.MODEL_SIZE} model metrics saved to {results_path}metrics_{hour}{minute}.csv")
    print(f"Evaluation on {config.MODEL_SIZE} model predictions/labels saved to {results_path}pred_lab_{hour}{minute}.json\n")

if __name__ == "__main__":
    main()
