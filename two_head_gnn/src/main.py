import os 
import torch
from torch_geometric.data import Data, DataLoader
from torch.utils.data import random_split
from sklearn.metrics import f1_score, accuracy_score
from taskGC_heterogenous import GraphHeterogenous
from TwoHeadGAT import TwoHeadGAT
import pandas as pd
from tqdm import tqdm
import datetime

month = datetime.datetime.now().month
day = datetime.datetime.now().day
year = datetime.datetime.now().year
hour = datetime.datetime.now().hour
minute = datetime.datetime.now().minute

# Load dataset and create graph        
def load_dataset(vision_path, llm_path, label_path, num_data_samples):
    data_list = [] # List for laoding all data samples (final len = num_data_samples)
    for d in range(num_data_samples):
        g_id = d
        print(d)
        vision_data = f"{vision_path}sample_{d}.json"
        llm_data = f"{llm_path}sample_{d}.json"
        label_data = f"{label_path}sample_{d}.json"
        
        graph = GraphHeterogenous(
            action_primitives=["pick", "insert", "lock", "putdown"],
            goal_states = ["insert", "lock"],
            vision_in=vision_data,
            llm_in=llm_data,
            label_in=label_data
        )
        
        graph.gen_encodings()
        wire_encodings = graph.get_wire_encodings()
        terminal_encodings = graph.get_terminal_encodings()

        x = torch.cat([wire_encodings, terminal_encodings.unsqueeze(0)])
        print(f"Length of x: {len(x)}")
        
        edge_index = graph.get_edge_index()
        edge_attr = graph.get_edge_attr()
        y_wire, y_action = graph.get_labels()
        print(f"wire label : {y_wire}")
        print(f"Action Label: {y_action}")
    
        
        data_list.append(Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y_wire=y_wire.unsqueeze(0),
            y_action=y_action.unsqueeze(0),
            wire_mask=graph.wire_mask,
            terminal_mask=graph.terminal_mask,
            graph_id=g_id
        )
                         )
        print(f"Processing Data: {d}")
        
    return data_list
    
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    epoch_wire_loss = 0
    epoch_action_loss = 0
    
    all_wire_preds, all_wire_labels = [], []
    all_action_preds, all_action_labels = [], []
    
    for data in loader:
        print(f"sample_{data.graph_id.item()}")
        # Node type masks
        wire_mask = data.wire_mask
        terminal_mask = data.terminal_mask
        
        # Extract global node indices or masked nodes
        wire_mask_idx = data.wire_mask.nonzero(as_tuple=False).squeeze()
        data = data.to(device)
        optimizer.zero_grad()
        
        wire_logits, action_logits = model(
            data.x.float(), wire_mask, data.edge_index, data.edge_attr, data.batch
        )
        print(torch.tensor(data.y_wire.squeeze()).to(device))
        wire_label =torch.tensor(data.y_wire.squeeze()).to(device)
        action_label = torch.tensor(data.y_action).float().to(device)
        if action_label.ndim == 2: # Shape [B, 4]
            action_label = action_label.argmax(dim=1)
        else: # Shape [4]
            action_label = action_label.unsqueeze(0).argmax(dim=1)
        
        print("Labels min:", wire_label.min().item())
        print("Labels max:", wire_label.max().item())
        print("Labels shape:", wire_label.shape)
        print(f"wire label : {wire_label}")
        print(f"wire predict: {wire_logits}")

        # Loss weights
        wire_weight = 2.0
        action_weight = 1.0
        print(wire_logits.dtype, wire_label.dtype)
        # Compute Loss
        wire_loss = criterion(wire_logits, wire_label)
        print(f"Wire Loss {wire_loss}")
        action_loss = criterion(action_logits, action_label)
        epoch_action_loss += action_loss.item()
        
        loss = (wire_weight * wire_loss) + (action_weight * action_loss)
        loss.backward()
        optimizer.step()
        
        epoch_wire_loss += wire_loss.item()
        epoch_action_loss += action_loss.item()
        total_loss += loss.item()
        
        # Predictions
        print(f"Prediciton: {wire_logits.argmax().item()}")
        print(f"With Mask: {wire_mask_idx[wire_logits.argmax().item()]}")
        wire_pred_global = wire_mask_idx[wire_logits.argmax().item()].item() 
        wire_label_global = wire_mask_idx[wire_label.item()].item()
        all_wire_preds.append(wire_pred_global)
        all_wire_labels.append(wire_label_global)
        all_action_preds.append(action_logits.argmax().item())
        all_action_labels.append(action_label.item())
        
    # Accuracy per head
    wire_acc = accuracy_score(all_wire_labels, all_wire_preds)
    action_acc = accuracy_score(all_action_labels, all_action_preds)
    
    # F1 Score
    wire_f1 =f1_score(all_wire_labels, all_wire_preds, average="weighted")
    action_f1 = f1_score(all_action_labels, all_action_preds, average="weighted")
    
    # Epoch Loss
    averaged_total_loss = total_loss / len(loader)
    averaged_wire_loss = epoch_wire_loss / len(loader)
    averaged_action_loss = epoch_action_loss / len(loader)
    
    return averaged_total_loss, wire_acc, wire_f1, averaged_wire_loss, action_acc, action_f1, averaged_action_loss       

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    epoch_val_wire_loss = 0
    epoch_val_action_loss = 0
    
    all_wire_preds, all_wire_labels = [], []
    all_action_preds, all_action_labels = [], []
    
    with torch.no_grad():
        for data in loader:
            print(f"sample_{data.graph_id.item()}")
            # Node type masks
            wire_mask = data.wire_mask
            terminal_mask = data.terminal_mask
            # Extract global node indices for masked nodes
            wire_mask_idx = data.wire_mask.nonzero(as_tuple=False).squeeze()
            
            data = data.to(device)
            
            wire_logits, action_logits = model(
                data.x.float(), wire_mask, data.edge_index, data.edge_attr, data.batch
            )    
            
            wire_label = torch.tensor(data.y_wire.item()).to(device)
            action_label = torch.tensor(data.y_wire.item()).to(device)
            print(action_label.shape)
            action_label = action_label.argmax(dim=1).long()
                        
            # Loss weights
            wire_weight = 2.0
            action_weight = 1.0
            
            wire_loss = criterion(wire_logits, wire_label)
            epoch_val_wire_loss += wire_loss.item()
            action_loss = criterion(action_logits, action_label)
            epoch_val_action_loss += action_loss.item()
            
            loss = (wire_weight * wire_loss) + (action_weight * action_loss)
            total_loss += loss.item()
            
            # Predictions
            wire_pred_global = wire_mask_idx[wire_logits.argmax().item()].item() 
            wire_label_global = wire_mask_idx[wire_label.item()].item()
            
            all_wire_preds.append(wire_pred_global)
            all_wire_labels.append(wire_label_global)
            all_action_preds.append(action_logits.argmax().item())
            all_action_labels.append(action_label.item())
            
    # Accuracy per head
    wire_val_acc = accuracy_score(all_wire_labels, all_wire_preds)
    action_val_acc = accuracy_score(all_action_labels, all_action_preds)
    
    # F1 Scores
    wire_val_f1 = f1_score(all_wire_labels, all_wire_preds, average="weighted")
    action_val_f1 = f1_score(all_action_labels, all_action_preds, average="weighted")
    
    # Epoch Loss
    averaged_val_loss = total_loss / len(loader)
    averaged_val_wire_loss = epoch_val_wire_loss / len(loader)
    averaged_val_action_loss = epoch_val_action_loss / len(loader)
    
    return averaged_val_loss, wire_val_acc, wire_val_f1, averaged_val_wire_loss, action_val_acc, action_val_f1, averaged_val_action_loss 

def main():
    training_results = {
        "epoch": [],
        "train_loss": [],
        "wire_train_acc": [],
        "wire_train_f1": [],
        "wire_train_loss": [],
        "action_train_acc": [],
        "action_train_f1": [],
        "action_train_loss": [],
        "val_loss": [],
        "wire_val_acc": [],
        "wire_val_loss": [],
        "wire_val_f1": [],
        "action_val_acc": [],
        "action_val_f1": [],
        "action_val_loss": []  
    }  
    
    vision_data = "../synthetic_data/4_class/vision/"
    llm_data = "../synthetic_data/4_class/llm/"
    label_data = "../synthetic_data/4_class/labels/"
    
    print("\n\nLoading Data...\n\n")
    
    dataset = load_dataset(vision_data, llm_data, label_data, num_data_samples=20)
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_data, val_data = random_split(dataset, [train_size, val_size], generator=generator)
    print(f"Length of train data / val data: {len(train_data)} / {len(val_data)}")    
    
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoHeadGAT(in_dim=len(dataset[0].x[0]), edge_feat_dim=1, hidden_dim=64, num_actions=4).to(device)
    
    checkpoint_path = "TwoHeadGAT_4class_0609.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_dict = model.state_dict()    
        pretrained_dict = {k:v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model = model.to(device)
        print(f"\n\nCheckpoint Loaded at {checkpoint_path}")
    else:
        print(f"\n\nCheckpoint not found at {checkpoint_path}. Training from Scratch.\n\n")
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Epoch Loop
    for epoch in tqdm(range(2), desc="training Epochs"):
        train_loss, wire_acc, wire_f1, wire_loss, action_acc, action_f1, action_loss = train(model, train_loader, optimizer, criterion, device=device)
        val_loss, wire_val_acc, wire_val_f1, wire_val_loss, action_val_acc, action_val_f1, action_val_loss = validate(model, val_loader, criterion, device=device)
        
        print(f"\nEpoch {epoch+1}:\nTraining loss {train_loss:.4f},\nWire Acc {wire_acc:.4f}, Wire F1 {wire_f1:.4f}, Wire Loss {wire_loss:.4f},\nAction Acc {action_acc:.4f}, Action F1 {action_f1:.4f}, Action Loss {action_loss:.4f}")
        print(f"\nEpoch {epoch+1}: \nValidation Loss {val_loss:.4f},\nWire Val Acc {wire_val_acc:.4f}, Wire Val F1 {wire_val_f1:.4f}, Wire Val Loss {wire_val_loss:.4f},\nAction Val Acc {action_val_acc:.4f}, Action Val F1 {action_val_f1:.4f}, Action Val Loss {action_val_loss:.4f}")
        training_results["epoch"].append(epoch)
        training_results["train_loss"].append(train_loss)
        training_results["wire_train_acc"].append(wire_acc)
        training_results["wire_train_f1"].append(wire_f1)
        training_results["wire_train_loss"].append(wire_loss)
        training_results["action_train_acc"].append(action_acc)
        training_results["action_train_f1"].append(action_f1)
        training_results["action_train_loss"].append(action_loss)
        training_results["val_loss"].append(val_loss)
        training_results["wire_val_acc"].append(wire_val_acc)  
        training_results["wire_val_f1"].append(wire_val_f1)
        training_results["wire_val_loss"].append(wire_val_loss) 
        training_results["action_val_acc"].append(action_val_acc)
        training_results["action_val_f1"].append(action_val_f1)
        training_results["action_val_loss"].append(action_val_loss) 
        
    # Save the model
    torch.save(model.state_dict(), "TwoHeaGAT_4class_0609.pth")   
    print("Model weights saved.")
    results_df = pd.DataFrame.from_dict(training_results)
    results_path = f"../docs/TwoHead_training_results/TwoHeadGAT_{year}_{month}_{day}"
    os.makedirs(results_path, exist_ok=True)
    results_df.to_csv(f"{results_path}_{hour}{minute}.csv")
    
if __name__ == "__main__":
    main()