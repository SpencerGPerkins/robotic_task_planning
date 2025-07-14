import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

BACKBONE_NAME = 'microsoft/deberta-v3-small'
MODEL_WEIGHTS_PATH = "deberta_multitask.pth"

NUM_WIRE_CLASSES = 5
NUM_TERMINAL_CLASSES = 11 

MAX_LENGTH = 128
BOTTLENECK_DIM =16

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
        
        logits_wire = self.head_wire(bottleneck_out)
        logits_terminal = self.head_terminal(bottleneck_out)
        
        return logits_wire, logits_terminal, bottleneck_out
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(BACKBONE_NAME)

model = DebertaMultiTask(
    backbone_name=BACKBONE_NAME,
    num_wire=NUM_WIRE_CLASSES,
    num_terminal=NUM_TERMINAL_CLASSES,
    bottleneck_dim=BOTTLENECK_DIM
)

model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device,weights_only=True))
        
model.to(device)

model.eval()

def predict_dual(text, model, tokenizer, device, max_len=128):
    
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits_wire, logits_terminal, bot = model(input_ids=input_ids, attention_mask=attention_mask)
    
    pred_wire = torch.argmax(logits_wire, dim=1).item()
    pred_terminal = torch.argmax(logits_terminal, dim=1).item()
    
    return pred_wire, pred_terminal, bot

def bert_muti_predict(text):
    predicted_wire_class, predicted_terminal_class, bot = predict_dual(text, model, tokenizer, device, MAX_LENGTH)
    
    print(f"Predicted Wire Class: {predicted_wire_class}"
          f"\nPredicted Terminal Class: {predicted_terminal_class}"
          f"\nBottleneck Output: {bot}")

bert_muti_predict("insert_black-wire_terminal_8")
bert_muti_predict("lock_green-wire_terminal_2")