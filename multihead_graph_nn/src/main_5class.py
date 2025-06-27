import os
import json
import torch
from torch_geometric.data import Data, DataLoader
from torch.utils.data import random_split
from sklearn.metrics import f1_score, accuracy_score
from graph_constructor_heterogenous import GraphHeterogenous
from  MultiheadGSGAT import MultiHeadGSGAT
import pandas as pd
from tqdm import tqdm
import datetime

month = datetime.datetime.now().month
day = datetime.datetime.now().day
year = datetime.datetime.now().year
hour = datetime.datetime.now().hour
minute = datetime.datetime.now().minute


# Load dataset and create a graph instance
def load_dataset(vision_path, llm_path, label_path, num_data_samples):
    data_list = [] # List for loading all data samples (final length = num_data_samples)
    for f in range(num_data_samples):
        print(f)
        vision_data = f"{vision_path}sample_{f}.json"
        llm_data = f"{llm_path}sample_{f}.json"
        label_data = f"{label_path}sample_{f}.json"
        
        graph = GraphHeterogenous(
            vision_in=vision_data,
            llm_in=llm_data,
            label_in=label_data
        )
        graph.gen_encodings()
        
        wire_encodings = graph.get_wire_encodings()
        wire_positions = graph.get_wire_positions()
        terminal_encodings = graph.get_terminal_encodings()
        terminal_positions = graph.get_terminal_positions()
        goal_encodings = graph.get_goal_encodings()
        
        x = torch.cat([wire_encodings, terminal_encodings, goal_encodings.unsqueeze(0)], dim=0)
        pos_x = torch.cat([wire_positions, terminal_positions], dim=0)
        
        edge_index = graph.get_edge_index()
        edge_attr = graph.get_edge_attr()
        
        y_wire, y_terminal, y_action = graph.get_labels()  # Wire ID, terminal ID, action (one-hot)
        
        data_list.append(Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            x_positions=pos_x,
            y_wire=y_wire.unsqueeze(0),
            y_terminal=y_terminal.unsqueeze(0),
            y_action=y_action.unsqueeze(0),
            wire_mask=graph.wire_mask,
            terminal_mask=graph.terminal_mask
        )
                         )
        print(f"Processing Data: {f}")
        
    return data_list

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    all_wire_preds, all_wire_labels = [], [] 
    all_terminal_preds, all_terminal_labels = [], []
    all_action_preds, all_action_labels = [], []
    
    for data in loader:
        # Node type masks
        wire_mask = data.wire_mask
        terminal_mask = data.terminal_mask

        # Extract global node indices for masked nodes
        wire_mask_idx = data.wire_mask.nonzero(as_tuple=False).squeeze()
        terminal_mask_idx = data.terminal_mask.nonzero(as_tuple=False).squeeze()
        data = data.to(device)
        optimizer.zero_grad()
        
        wire_logits, terminal_logits, action_logits = model(
            data.x.float(), wire_mask, terminal_mask, data.edge_index, data.edge_attr, data.batch
        )
        wire_label = torch.tensor(data.y_wire.item()).to(device)
        terminal_label = torch.tensor(data.y_terminal.item()).to(device)
        
        # action_label = torch.tensor(data.y_action).float().to(device)
        # action_label = action_label.argmax(dim=1).long()
        action_label = torch.tensor(data.y_action).float().to(device)
        if action_label.ndim == 2:  # shape [B, 5]
            action_label = action_label.argmax(dim=1)
        else:  # shape [5]
            action_label = action_label.unsqueeze(0).argmax(dim=1)


        # Loss weights
        wire_weight = 2.0
        terminal_weight = 1.0
        action_weight = 3.0

        # Compute Loss
        wire_loss = criterion(wire_logits, wire_label)
        terminal_loss = criterion(terminal_logits, terminal_label)
        action_loss = criterion(action_logits, action_label)
        
        loss = (wire_weight * wire_loss) + (terminal_weight * terminal_loss) + (action_weight * action_loss)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Predictions
        wire_pred_global = wire_mask_idx[wire_logits.argmax().item()].item()
        wire_label_global = wire_mask_idx[wire_label.item()].item()
        terminal_pred_global = terminal_mask_idx[terminal_logits.argmax().item()].item()
        terminal_label_global = terminal_mask_idx[terminal_label.item()].item()

        all_wire_preds.append(wire_pred_global)
        all_wire_labels.append(wire_label_global)
        all_terminal_preds.append(terminal_pred_global)
        all_terminal_labels.append(terminal_label_global)
        all_action_preds.append(action_logits.argmax().item())
        all_action_labels.append(action_label.item())        
        
    # Accuracy per head
    wire_acc = accuracy_score(all_wire_labels, all_wire_preds)
    terminal_acc = accuracy_score(all_terminal_labels, all_terminal_preds)
    action_acc = accuracy_score(all_action_labels, all_action_preds)
    
    # F1 Score
    wire_f1 = f1_score(all_wire_labels, all_wire_preds, average="weighted")
    terminal_f1 = f1_score(all_terminal_labels, all_terminal_preds, average="weighted")
    action_f1 = f1_score(all_action_labels, all_action_preds, average="weighted")
    
    return total_loss /len(loader), wire_acc, wire_f1, terminal_acc, terminal_f1, action_acc, action_f1     

def validate(model, loader, criterion, device):
    
    model.eval()
    total_loss = 0
    
    all_wire_preds, all_wire_labels = [], []
    all_terminal_preds, all_terminal_labels = [], []
    all_action_preds, all_action_labels = [], []
    
    with torch.no_grad():
        for data in loader:
            # Node type masks
            wire_mask = data.wire_mask
            terminal_mask = data.terminal_mask
            # Extract global node indices for masked nodes
            wire_mask_idx = data.wire_mask.nonzero(as_tuple=False).squeeze()
            terminal_mask_idx = data.terminal_mask.nonzero(as_tuple=False).squeeze()
            
            data = data.to(device)
            
            wire_logits, terminal_logits, action_logits = model(
                data.x.float(), wire_mask, terminal_mask, data.edge_index, data.edge_attr, data.batch
            )
            
            wire_label = torch.tensor(data.y_wire.item()).to(device)
            terminal_label = torch.tensor(data.y_terminal.item()).to(device)

            action_label = torch.tensor(data.y_action).float().to(device)
            action_label = action_label.argmax(dim=1).long()       

            # Loss weights
            wire_weight = 2.0
            terminal_weight = 1.0
            action_weight = 3.0
            
            wire_loss = criterion(wire_logits, wire_label)
            terminal_loss = criterion(terminal_logits, terminal_label)
            action_loss = criterion(action_logits, action_label)
            
            loss = (wire_weight * wire_loss) + (terminal_weight * terminal_loss) + (action_weight * action_loss)
            total_loss += loss.item()
            
            # Predictions
            wire_pred_global = wire_mask_idx[wire_logits.argmax().item()].item()
            wire_label_global = wire_mask_idx[wire_label.item()].item()
            terminal_pred_global = terminal_mask_idx[terminal_logits.argmax().item()].item()
            terminal_label_global = terminal_mask_idx[terminal_label.item()].item()

            all_wire_preds.append(wire_pred_global)
            all_wire_labels.append(wire_label_global)
            all_terminal_preds.append(terminal_pred_global)
            all_terminal_labels.append(terminal_label_global)
            all_action_preds.append(action_logits.argmax().item())
            all_action_labels.append(action_label.item())  
            
    # Accuracy per head
    wire_val_acc = accuracy_score(all_wire_labels, all_wire_preds)
    terminal_val_acc = accuracy_score(all_terminal_labels, all_terminal_preds)
    action_val_acc = accuracy_score(all_action_labels, all_action_preds)
    
    # F1 Scores
    wire_val_f1 = f1_score(all_wire_labels, all_wire_preds, average="weighted")
    terminal_val_f1 = f1_score(all_terminal_labels, all_terminal_preds, average="weighted")
    action_val_f1 =f1_score(all_action_labels, all_action_preds, average="weighted")
    
    return total_loss / len(loader), wire_val_acc, wire_val_f1, terminal_val_acc, terminal_val_f1, action_val_acc, action_val_f1

def main():
    training_results = {
        "epoch": [],
        "train_loss": [],
        "wire_train_acc": [],
        "wire_train_f1": [],
        "terminal_train_acc": [],
        "terminal_train_f1": [],
        "action_train_acc": [],
        "action_train_f1": [],
        "val_loss": [],
        "wire_val_acc":[],
        "wire_val_f1": [],
        "terminal_val_acc": [],
        "terminal_val_f1": [],
        "action_val_acc": [],
        "action_val_f1": []        
    }           
    vision_data= f"../synthetic_data/5_class/vision/"
    llm_data = f"../synthetic_data/5_class/llm/"
    label_data = f"../synthetic_data/5_class/labels/"
    
    print("\n\nLoading Data...\n\n")
    
    dataset = load_dataset(vision_data, llm_data, label_data, num_data_samples=200)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_data, val_data = random_split(dataset, [train_size, val_size], generator=generator)
    print(f"Length of train data / val data: {len(train_data)} / {len(val_data)}")
    
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiHeadGSGAT(in_dim=len(dataset[0].x[0]), edge_feat_dim=1, hidden_dim=64, num_actions=5).to(device)
    
    checkpoint_path = "MultiHead_GSGAT_5class.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("\n\nCheckpoint Loaded.\n\n")
    else:
        print("\n\nCheckpoint not found. Training from scratch.\n\n")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in tqdm(range(200), desc="Training Epochs"):
        train_loss, wire_acc, wire_f1, terminal_acc, terminal_f1, action_acc, action_f1 = train(model, train_loader, optimizer, criterion, device=device)
        val_loss, wire_val_acc, wire_val_f1, terminal_val_acc, terminal_val_f1, action_val_acc, action_val_f1 = validate(model, val_loader, criterion, device=device)
        
        print(f"\nEpoch {epoch+1}: Training Loss {train_loss:.4f}, Wire Acc {wire_acc:.4f}, Wire F1 {wire_f1:.4f}, Terminal Acc {terminal_acc}, Terminal F1 {terminal_f1:.4f}, Action Acc {action_acc:.4f}, Action F1 {action_f1:.4f}")
        print(f"Epoch {epoch+1}: Validation Loss {val_loss:.4f}, Wire Validation Acc {wire_val_acc:.4f}, Wire Validation F1 {wire_val_f1:.4f}, Terminal Validation Acc {terminal_val_acc}, Terminal Validation F1 {terminal_val_f1:.4f}, Action Validation Acc {action_val_acc:.4f}, Action Validation F1 {action_val_f1:.4f}\n")
        training_results["epoch"].append(epoch)
        training_results["train_loss"].append(train_loss)
        training_results["wire_train_acc"].append(wire_acc)
        training_results["wire_train_f1"].append(wire_f1)
        training_results["terminal_train_acc"].append(terminal_acc)
        training_results["terminal_train_f1"].append(terminal_f1)
        training_results["action_train_acc"].append(action_acc)
        training_results["action_train_f1"].append(action_f1)
        training_results["val_loss"].append(val_loss)
        training_results["wire_val_acc"].append(wire_val_acc)
        training_results["wire_val_f1"].append(wire_val_f1)
        training_results["terminal_val_acc"].append(terminal_val_acc)
        training_results["terminal_val_f1"].append(terminal_val_f1)
        training_results["action_val_acc"].append(action_val_acc)
        training_results["action_val_f1"].append(action_val_f1)        
        
    # Save the trained model
    torch.save(model.state_dict(), "MultiHead_GSGAT_5class_ckpt1.pth")
    print("Model weights saved.")    
    results_df = pd.DataFrame.from_dict(training_results)
    results_path = f"../docs/multihead_training_results/MultiHeadGSGAT_{year}_{month}_{day}"
    results_df.to_csv(f"{results_path}_{hour}{minute}.csv")
    
if __name__ == "__main__":
    main()    
        
    

    
    
    
    
                        
            
        
        
        
        