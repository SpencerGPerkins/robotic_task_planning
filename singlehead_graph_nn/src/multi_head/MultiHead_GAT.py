import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv, global_mean_pool

class MultiHeadGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, max_wires, max_terminals, num_actions, heads=4, dropout=0.6):
        super(MultiHeadGAT, self).__init__()
        
        # First layer: GraphSAGE for efficient sampling-based propagation
        self.sage_conv = SAGEConv(in_dim, hidden_dim)
        
        # Second layer: GAT for attention-based aggregation
        self.gat_conv = GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout, concat=False)
        
        # Output heads for binary classification (sigmoid output for binary classification)
        self.wire_head = nn.Linear(hidden_dim, 1)  # Binary classification, single output per node
        self.terminal_head = nn.Linear(hidden_dim, 1)  # Binary classification, single output per node
        
        # Global pooling layer (e.g., mean pooling)
        self.global_pool = global_mean_pool  

        self.action_head = nn.Linear(hidden_dim, num_actions)  

    def forward(self, x, edge_index, batch):
        # Apply GAT layers with edge attention
        x = F.elu(self.gat1(x, edge_index))
        x = F.elu(self.gat2(x, edge_index))
        
        num_wires = len(x) -10
        num_terminals = 10
        # Wire head: Binary classification (sigmoid activation)
        wire_logits = self.wire_head(x)[:, :num_wires]  # Only use relevant logits
        wire_probs = torch.sigmoid(wire_logits)  # Apply sigmoid for binary classification
        
        # Terminal head: Binary classification (sigmoid activation)
        terminal_logits = self.terminal_head(x)[:, num_wires:num_terminals]
        terminal_probs = torch.sigmoid(terminal_logits)  # Apply sigmoid for binary classification
        
        x = self.global_pool(x, batch)
        
        # Action head: Multi-class classification (softmax or other appropriate activation)
        action_logits = self.action_head(x)
        
        return wire_probs, terminal_probs, action_logits

