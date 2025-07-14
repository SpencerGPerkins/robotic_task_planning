import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

# after yrained by using train_debert.py you can use it

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        bert_hidden_size = self.bert.config.hidden_size
        bottleneck_dim = 32
        self.bottleneck = nn.Linear(bert_hidden_size, bottleneck_dim)
        self.fc = nn.Linear(bottleneck_dim, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        bottleneck_output = self.bottleneck(pooled_output)
        logits = self.fc(bottleneck_output)
        return logits, bottleneck_output

bert_model_name = 'microsoft/deberta-v3-small'
num_classes = 9
model_weights_path = "deberta_action.pth"
model = BERTClassifier(bert_model_name, num_classes)
model.load_state_dict(torch.load(model_weights_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() 
tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

def bert_predict(text):
    with torch.no_grad():
        encoding = tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        outputs, bot = model(input_ids=input_ids, attention_mask=attention_mask)
        _, predicted_class_id_tensor = torch.max(outputs, dim=1)
        predicted_class_id = predicted_class_id_tensor.item()
    print(predicted_class_id)
    print(bot)
    return predicted_class_id,bot

bert_predict("(lock_blue-wire_ter-4)")

