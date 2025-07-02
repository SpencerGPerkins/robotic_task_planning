import torch
from torch_geometric.data import DataLoader
from torch.utils.data import random_split
from datetime import datetime
import os
from tqdm import tqdm
import pandas as pd

import config
from data_process import load_dataset
from models import TwoHeadGAT, TwoHeadGATSmall
from utils import load_checkpoint
from train_val import train, validate

month = datetime.now().month
day = datetime.now().day
year = datetime.now().year
hour = datetime.now().hour
minute = datetime.now().minute

def main():
    torch.autograd.set_detect_anomaly(True)

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

    vision_data = config.VISION_DATA_PATH
    llm_data = config.LLM_DATA_PATH
    label_data = config.LABEL_DATA_PATH
    num_samples = config.NUM_SAMPLES

    print("\n\nLoading Data...\n\n")

    dataset = load_dataset(vision_data, llm_data, label_data, num_data_samples=num_samples, action_primitives=config.ACTION_PRIMS)
    train_size =int(0.8 * num_samples)
    val_size = num_samples - train_size
    generator = torch.Generator().manual_seed(42)
    train_data, val_data = random_split(dataset, [train_size, val_size], generator=generator)

    print(f"Length of train data / val data: {len(train_data)} / {len(val_data)}")    
    
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    device = config.DEVICE
    if config.MODEL_SIZE == "small":
        model = TwoHeadGATSmall(in_dim=len(dataset[0].x[0]), edge_feat_dim=1, hidden_dim=config.HIDDEN_DIM, num_actions=config.NUM_ACTIONS).to(device)
        print("\n\nSmall Model Used...\n\n")
    else:
        model = TwoHeadGAT(in_dim=len(dataset[0].x[0]), edge_feat_dim=1, hidden_dim=config.HIDDEN_DIM, num_actions=config.NUM_ACTIONS).to(device)
        print("\n\nMedium Model Used...\n\n")

    print("Exists:", os.path.exists(config.CHECKPOINT_PATH))
    if os.path.exists(config.CHECKPOINT_PATH):
        model = load_checkpoint(config.CHECKPOINT_PATH, model, torch.load)
    else:
        print(f"\n\nCheckpoint not found at {config.CHECKPOINT_PATH}. Training from Scratch.\n\n")        

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    # Epoch loop
    for epoch in tqdm(range(config.NUM_EPOCHS), desc="Training Epochs"):
        train_loss, wire_acc, wire_f1, wire_loss, action_acc, action_f1, action_loss = train(model, train_loader, optimizer, criterion, device=device)
        val_loss, wire_val_acc, wire_val_f1, wire_val_loss, action_val_acc, action_val_f1, action_val_loss = validate(model, val_loader, criterion, device=device)
        
        print(f"\nEpoch {epoch+1}:\nTraining loss {train_loss:.4f},\nWire Acc {wire_acc:.4f}, Wire F1 {wire_f1:.4f}, Wire Loss {wire_loss:.4f},\nAction Acc {action_acc:.4f}, Action F1 {action_f1:.4f}, Action Loss {action_loss:.4f}")
        print(f"--------------- \nValidation Loss {val_loss:.4f},\nWire Val Acc {wire_val_acc:.4f}, Wire Val F1 {wire_val_f1:.4f}, Wire Val Loss {wire_val_loss:.4f},\nAction Val Acc {action_val_acc:.4f}, Action Val F1 {action_val_f1:.4f}, Action Val Loss {action_val_loss:.4f}")
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
    torch.save(model.state_dict(), config.SAVE_MODEL_WEIGHTS)   
    print("Model weights saved.")
    results_df = pd.DataFrame.from_dict(training_results)
    results_path = f"{config.SAVE_RESULTS_HEAD}_{year}_{month}_{day}/"
    os.makedirs(results_path, exist_ok=True)
    results_df.to_csv(f"{results_path}_{hour}{minute}_{config.DATASET}.csv")
    print(f"Training/Validation Completed on {config.DATASET} Dataset")
    
if __name__ == "__main__":
    main()        


    