import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv, SAGEConv, NNConv, global_mean_pool

class GraphSAGE_GAT(torch.nn.Module):
    def __init__(self, in_dim, edge_feat_dim, hidden_dim, num_actions, heads=4, dropout=0.6):
        super(GraphSAGE_GAT, self).__init__()
        
        # Use NNConv to inject edge features early
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_feat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, in_dim * in_dim)
        )
        self.nn_conv = NNConv(in_dim, in_dim, self.edge_encoder, aggr='mean')
        
        # First layer: GraphSAGE for efficient sampling-based propagation
        self.sage_conv = SAGEConv(in_dim, hidden_dim)
        
        # Second layer: GAT for attention-based aggregation
        self.gat_conv = GATConv(hidden_dim, hidden_dim, heads=heads, dropout=dropout, concat=False)
        
        self.dropout = dropout
        
        # Global pooling layer (e.g., mean pooling)
        self.global_pool = global_mean_pool  

        self.action_head = nn.Linear(hidden_dim, num_actions)  
        
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.nn_conv(x, edge_index, edge_attr)
        x = F.relu(x)
        
        # Step 1: GraphSAGE Layer
        x = self.sage_conv(x, edge_index)
        x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Step 2: GAT Layer
        x = self.gat_conv(x, edge_index)
        x = F.relu(x)

        x = self.global_pool(x, batch)

        action_logits = self.action_head(x)
        
        return action_logits


