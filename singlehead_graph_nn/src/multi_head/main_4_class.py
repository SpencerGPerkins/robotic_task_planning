import os
import json
import torch
from torch_geometric.data import Data, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from graph_constructor_partial import GraphCategorical
from  MultiHead_GAT import MultiHeadGAT
import pandas as pd

# Load dataset and create graph instances
def load_dataset(vision_path, llm_path, label_path, num_data_samples):
    data_list = []
    for f in range(num_data_samples):
        vision_data = f"{vision_path}sample_{f}.json"
        llm_data = f"{llm_path}sample_{f}.json"
        label_data = f"{label_path}sample_{f}.json"
        
        # Create Graph instance
        graph = GraphCategorical(
            vision_in=vision_data,
            llm_in=llm_data,
            label_in=label_data
        )
        graph.gen_encodings()
        encodings = graph.get_wire_encodings()
        
        print(type(encodings), encodings.layout)
        # Convert to PyG Data object
        wire_encodings = graph.get_wire_encodings()
        print(f"wire_encodings: {wire_encodings}")
        
        # padded_wire_encodings = [feat + torch.tensor([0]) * (13 - len(feat)) for feat in wire_encodings]
        terminal_encodings = graph.get_terminal_encodings()
        print(f"terminal encodings : {terminal_encodings}")
        # x = torch.tensor(list(wire_encodings) + list(terminal_encodings), dtype=torch.long)
        x = torch.cat([wire_encodings, terminal_encodings], dim=0)
        print(f"X : {x}")
        edge_index = graph.get_edge_index()
        print(f"Shape of edge index: {edge_index.shape}")
        y = graph.get_labels()
        # y = graph.get_labels()
        # y = torch.tensor(y).view(1, -1)  # Ensure y is [1, num_actions]
        print(f"Labels reshaped per sample: {y.shape}")  # Should be [1, 3]

        data_list.append(Data(x=x, edge_index=edge_index, y=y.unsqueeze(0)))
        print(f"Labels  : {y}")
        
        # data_list.append(Data(x=x, edge_index=edge_index, y=y))
        print(f"Processing data : {f}")
    return data_list
    
def reshape_action_labels(action_labels, batch_size):
    # Reshape the continuous list into a 2D tensor of shape [batch_size, 3]
    action_labels_reshaped = action_labels.view(batch_size, 3)
    return action_labels_reshaped

def train(model, loader, optimizer,criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for data in loader:
        print(f"Batch size: {data.x.size(0)}")  # This should print 20
        print(data)
        data = data.to(device)
        optimizer.zero_grad()
        
        # action_logits = model(
        #     data.x.float(), data.edge_index, data.batch,
        #     num_wires=len(data.x) - 10,  # Assuming last 10 nodes are terminals
        #     num_terminals=10
        # )
        action_logits = model(
            data.x.float(), data.edge_index, data.batch
        )

        # Unpack labels
        print(f"data.y: {data.y}")
        action_label = torch.tensor(data.y).float()
        
        # Reshape action labels to match logits shape
        batch_size = len(data.x)  # Assuming batch_size is the number of nodes
        print(batch_size)
        # action_label = reshape_action_labels(action_label, batch_size)
        # action_label = action_label.argmax(dim=1) if action_label.dim() > 1 else action_label
        # print(f"Action logits shape : {action_logits.shape}")
        # print(f"Action Label shape: {action_label.shape}")
        # print(f"Action labels : {action_label}")
        # For action, apply CrossEntropyLoss (multi-class classification)
        action_loss = criterion(action_logits, action_label)

        # Total loss
        loss = action_loss
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # For action classification, get predicted class
        all_preds.extend(action_logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(action_label.argmax(dim=1).cpu().numpy())
    print(f"all labels: {all_labels}")
    print(f"all_preds: {all_preds}")
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return total_loss / len(loader), acc, f1

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            # action_logits = model(
            #     data.x.float(), data.edge_index,data.batch,
            #     num_wires=len(data.x) - 10,
            #     num_terminals=10
            # )
            action_logits = model(
                data.x.float(), data.edge_index, data.batch
            )


        # Unpack labels
        print(f"data.y: {data.y}")
        action_label = torch.tensor(data.y).float()
        
        # Reshape action labels to match logits shape
        batch_size = len(data.x)  # Assuming batch_size is the number of nodes
        print(batch_size)
        # action_label = reshape_action_labels(action_label, batch_size)
        # action_label = action_label.argmax(dim=1) if action_label.dim() > 1 else action_label
        # print(f"Action logits shape : {action_logits.shape}")
        # print(f"Action Label shape: {action_label.shape}")
        # print(f"Action labels : {action_label}")
        # For action, apply CrossEntropyLoss (multi-class classification)
        action_loss = criterion(action_logits, action_label)

        # Total loss
        loss = action_loss
    
        total_loss += loss.item()
        
        # For action classification, get predicted class
        all_preds.extend(action_logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(action_label.argmax(dim=1).cpu().numpy())
    print(f"all labels: {all_labels}")
    print(f"all_preds: {all_preds}")
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return total_loss / len(loader), acc, f1


def main():
    # Main script
    training_results = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "train_f1": [],
        "val_loss":[],
        "val_acc": [],
        "val_f1": []
        }
    vision_data = f"../synthetic_data/4_class/vision/"
    llm_data = f"../synthetic_data/4_class/llm/"
    label_data = f"../synthetic_data/4_class/labels/"
    dataset = load_dataset(vision_data, llm_data, label_data, num_data_samples=200)
    print(f"dataset: {dataset}")
    dataset_size = len(dataset)
    train_size = int(0.8 * dataset_size)
    train_data, val_data = dataset[:train_size], dataset[train_size:]
    print(f"len of train data: {len(train_data)}")

    train_loader = DataLoader(train_data, batch_size=20, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load("model_weights.pth", map_location="cpu")
    
    model = MultiHeadGAT(in_dim=len(dataset[0].x[0]), hidden_dim=16, num_actions=4).to(device)

    # Load only matching parameters
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    with torch.no_grad():
        model.action_head.weight[-1].copy_(torch.randn_like(model.action_head.weight[0]))
        model.action_head.bias[-1].copy_(torch.tensor(0.0))  # Adjust if needed

    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    action_criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(500):
        train_loss, train_acc, train_f1 = train(model, train_loader, optimizer, action_criterion,  device=device)
        val_loss, val_acc, val_f1 = validate(model, val_loader,  action_criterion, device=device)
        print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Acc {train_acc:.4f}, F1 {train_f1:.4f} | Val Loss {val_loss:.4f}, Acc {val_acc:.4f}, F1 {val_f1:.4f}")
        training_results["epoch"].append(epoch)
        training_results["train_loss"].append(train_loss)
        training_results["train_acc"].append(train_acc)
        training_results["train_f1"].append(train_f1)
        training_results["val_loss"].append(val_loss)
        training_results["val_acc"].append(val_acc)
        training_results["val_f1"].append(val_f1)
        
        
    # Save the trained model
    torch.save(model.state_dict(), "GraphSAGE_GAT_MultiHead_model_weights_4_class.pth")
    print("Model weights saved.")    
    results_df = pd.DataFrame.from_dict(training_results)
    results_df.to_csv("../docs/training_results/graph_sage_0321.csv")
    
if __name__ == "__main__":
    main()

