import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BACKBONE_NAME = 'microsoft/deberta-v3-small'
DATA_FILE = "gai_merged_wire_terminal.csv" # train file here
MODEL_OUTPUT_PATH = "deberta_multitask.pth"

MAX_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 15 
LEARNING_RATE = 2e-5
BOTTLENECK_DIM = 16 # Dimension be careful

class MultiTaskDataset(Dataset):
    def __init__(self, texts, labels_wire, labels_terminal, tokenizer, max_len):
        self.texts = texts
        self.labels_wire = labels_wire
        self.labels_terminal = labels_terminal
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label_wire': torch.tensor(self.labels_wire[idx], dtype=torch.long),
            'label_terminal': torch.tensor(self.labels_terminal[idx], dtype=torch.long),
        }

class DebertaMultiTask(nn.Module):
    def __init__(self, backbone_name, num_wire, num_terminal, bottleneck_dim):
        super().__init__()
        self.bert = AutoModel.from_pretrained(backbone_name)
        hidden_size = self.bert.config.hidden_size 
        
        self.bottleneck = nn.Linear(hidden_size, bottleneck_dim)
        
        self.head_wire = nn.Linear(bottleneck_dim, num_wire)
        self.head_terminal = nn.Linear(bottleneck_dim, num_terminal)
        
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]
        
        bottleneck_out = self.bottleneck(pooled) 
        bottleneck_out = self.relu(bottleneck_out)
        bottleneck_out = self.drop(bottleneck_out)
        
        # Get logits from each task head
        logits_wire = self.head_wire(bottleneck_out)
        logits_terminal = self.head_terminal(bottleneck_out)
        
        return logits_wire, logits_terminal

def train_one_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model.train()
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        y_wire = batch['label_wire'].to(device)
        y_terminal = batch['label_terminal'].to(device)

        logits_wire, logits_terminal = model(input_ids, attention_mask)

        # Calculate loss for each task (ignore_index handles -1 labels)
        loss_wire = loss_fn(logits_wire, y_wire)
        loss_terminal = loss_fn(logits_terminal, y_terminal)

        # Combine losses (simple sum)
        loss = loss_wire + loss_terminal
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)

def evaluate(model, data_loader, device):
    model.eval()
    g_wire, g_terminal = [], []
    p_wire, p_terminal = [], []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            y_wire = batch['label_wire'].to(device)
            y_terminal = batch['label_terminal'].to(device)

            logits_wire, logits_terminal = model(input_ids, attention_mask)

            # Filter out ignored labels (-1) before calculating metrics
            valid_wire_indices = y_wire != -1
            valid_terminal_indices = y_terminal != -1

            p_wire.extend(torch.argmax(logits_wire[valid_wire_indices], dim=1).cpu().tolist())
            g_wire.extend(y_wire[valid_wire_indices].cpu().tolist())
            
            p_terminal.extend(torch.argmax(logits_terminal[valid_terminal_indices], dim=1).cpu().tolist())
            g_terminal.extend(y_terminal[valid_terminal_indices].cpu().tolist())

    print("--- Validation Report: Wire ---")
    print(classification_report(g_wire, p_wire, zero_division=0))
    
    print("--- Validation Report: Terminal ---")
    print(classification_report(g_terminal, p_terminal, zero_division=0))
    
    return accuracy_score(g_wire, p_wire), accuracy_score(g_terminal, p_terminal)

if __name__ == "__main__":

    df = pd.read_csv(DATA_FILE)

    texts = df['goal'].tolist()
    labels_wire = df['wire'].tolist()
    labels_terminal = df['terminal'].tolist()

    # Determine number of classes for each task (ignoring the -1 label)
    num_wire_classes = df['wire'].max() + 1
    num_terminal_classes = df['terminal'].max() + 1
    print(f"Task 'wire' has {num_wire_classes} classes.")
    print(f"Task 'terminal' has {num_terminal_classes} classes.")

    train_idx, val_idx = train_test_split(range(len(texts)), test_size=0.2, random_state=32)

    train_texts = [texts[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    train_labels_wire = [labels_wire[i] for i in train_idx]
    val_labels_wire = [labels_wire[i] for i in val_idx]
    train_labels_terminal = [labels_terminal[i] for i in train_idx]
    val_labels_terminal = [labels_terminal[i] for i in val_idx]


    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_NAME)
    train_dataset = MultiTaskDataset(train_texts, train_labels_wire, train_labels_terminal, tokenizer, MAX_LENGTH)
    val_dataset = MultiTaskDataset(val_texts, val_labels_wire, val_labels_terminal, tokenizer, MAX_LENGTH)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DebertaMultiTask(
        backbone_name=BACKBONE_NAME,
        num_wire=num_wire_classes,
        num_terminal=num_terminal_classes,
        bottleneck_dim=BOTTLENECK_DIM
    ).to(device)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-1) # Ignores -1 labels
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_dataloader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    print(f"Starting training on {device}...")

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
        avg_loss = train_one_epoch(model, train_dataloader, loss_fn, optimizer, device, scheduler)
        print(f"Average Training Loss: {avg_loss:.4f}")
        acc_wire, acc_terminal = evaluate(model, val_dataloader, device)
        print(f"Validation Accuracy -> Wire: {acc_wire:.4f}, Terminal: {acc_terminal:.4f}")

    torch.save(model.state_dict(), MODEL_OUTPUT_PATH)
    print(f"\nTraining complete. Model saved to {MODEL_OUTPUT_PATH}")