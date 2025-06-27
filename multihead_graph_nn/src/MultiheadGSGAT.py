import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, NNConv, global_mean_pool

class MultiHeadGSGAT(nn.Module):
    def __init__(self, in_dim, edge_feat_dim, hidden_dim, num_actions, heads=4, dropout=0.3):
        super(MultiHeadGSGAT, self).__init__()

        # Injet edge features
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, in_dim * in_dim)
        )
        self.nn_conv = NNConv(in_dim, in_dim, self.edge_encoder, aggr='mean')

        # GraphSAGE layers
        self.sage_conv1 = SAGEConv(in_dim, hidden_dim)
        self.sage_conv2 = SAGEConv(hidden_dim, hidden_dim)
        
        # GAT layers
        self.gat_conv1 = GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout, concat=False)
        self.gat_conv2 = GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout, concat=False)
        
        self.dropout = dropout
        
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Heads
        self.wire_head = nn.Linear(hidden_dim, 1)
        self.terminal_head = nn.Linear(hidden_dim, 1)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, x, wire_mask, terminal_mask, edge_index, edge_attr, batch):
        x = self.nn_conv(x, edge_index, edge_attr)
        x = F.relu(x)

        # GraphSAGE Block 
        x1 = self.sage_conv1(x, edge_index)
        x1 = F.relu(x1)
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x1 = self.ln1(x1)
        
        x2 = self.sage_conv2(x1, edge_index) + x1
        x2 = F.relu(x2)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x2 = self.ln2(x2)

        # GAT Block
        x3 = self.gat_conv1(x2, edge_index)
        x3 = F.relu(x3)
        x4 = self.gat_conv2(x3, edge_index) + x3
        x4 = F.relu(x4)
        
        # # Node-level predictions
        p_wire = self.wire_head(x4[wire_mask]).squeeze(-1)
        p_terminal = self.terminal_head(x4[terminal_mask]).squeeze(-1)
        # p_wire = self.wire_head(x).squeeze(-1)
        # p_wire = p_wire[wire_mask]
        
        # p_terminal = self.terminal_head(x).squeeze(-1)
        # p_terminal = p_terminal[terminal_mask]
        
        

        # Graph-level prediction
        x_pooled = global_mean_pool(x4, batch)
        p_action = self.action_head(x_pooled)

        return p_wire, p_terminal, p_action
