# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class SpatialPositionalEncoding(nn.Module):
    
    def __init__(self, d_model, in_dim=6):
        super(SpatialPositionalEncoding, self).__init__()
        assert d_model % in_dim == 0,  'Dimensionality must be divisible by input dimensions X 2 **e.g. ((x,y,z) * 2)'
        self.d_model = d_model
        self.in_dim = in_dim
        self.F = d_model // (2 * self.in_dim) # Number of frequency pairs 
        self.freqs = torch.exp(
            torch.arange(0, self.F, dtype=torch.float32) * (-math.log(10000.0) / self.F)
        ).view(1, 1, 1, -1) # shape [1, 1, 1, F] for input [B, N, in_dim, 1]
        
    def forward(self, coords):
        batch_size, N, _ = coords.shape
        coords = coords.unsqueeze(-1) # [B, N, in_dim, 1]
        freqs = self.freqs.to(coords.device) # Number of frequency paires, self.F
        
        angles = coords * freqs #[B, N, 3, self.F]
        
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        
        # Concatenate sin and cos for each coord dimension
        pe = torch.cat([sin, cos], dim=-1) # [B, N, 3, d_model//in_dim]
        pe = pe.view(batch_size, N, self.d_model) # [B, N, d_model]
        return pe   
    

model = SpatialPositionalEncoding(d_model=60, in_dim=3)
t_0 = torch.tensor([0.32954907417297363, 0.4249360263347626, 0.00251], dtype=torch.float64) * 10
t_1 = torch.tensor([0.32954907417297363, 0.4249360263347626, 0.076], dtype=torch.float64) * 10
# t_0 = torch.tensor([0.3447992205619812,0.4457240402698517, 0.00251], dtype=torch.float64)
# t_1 = torch.tensor([0.32954907417297363, 0.4249360263347626, 0.076], dtype=torch.float64)

pe1 = model(t_0.unsqueeze(0).unsqueeze(0))
pe2 = model(t_1.unsqueeze(0).unsqueeze(0))



l2_diff = torch.norm(pe1 - pe2, p=2).item()
cos_diff = 1 - torch.nn.functional.cosine_similarity(pe1, pe2, dim=-1).item()

print(f"L2 Distance: {l2_diff}")
print(f"Cosine Distance: {cos_diff}")

# Extract vectors from batch
pe1_vec = pe1[0, 0].detach().cpu().numpy()
pe2_vec = pe2[0, 0].detach().cpu().numpy()

# Plot side-by-side line plots
plt.figure(figsize=(12, 5))
plt.plot(pe1_vec, label='Encoding at t0')
plt.plot(pe2_vec, label='Encoding at t1')
plt.title('Positional Encoding Features')
plt.xlabel('Feature Index')
plt.ylabel('Value')
plt.legend()
plt.show()