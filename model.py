from torch_geometric.nn import NNConv, Set2Set
import torch
from torch.nn import Module, Linear, ReLU, Sequential
from torch_geometric.nn import GCNConv, GATConv, NNConv, Set2Set, global_mean_pool

class GNNModel(Module):
    def __init__(self, num_node_features=20, hidden_dim=64, model_type='GCN'):
        super(GNNModel, self).__init__()
        self.model_type = model_type
        if model_type == 'GCN':
            self.conv1 = GCNConv(num_node_features, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif model_type == 'GAT':
            self.conv1 = GATConv(num_node_features, hidden_dim)
            self.conv2 = GATConv(hidden_dim, hidden_dim)
        elif model_type == 'MPNN':
            nn = Sequential(Linear(num_node_features, hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim * hidden_dim))
            self.conv = NNConv(num_node_features, hidden_dim, nn, aggr='mean')
            self.set2set = Set2Set(hidden_dim, processing_steps=3)
        else:
            raise ValueError("Invalid model type. Choose from 'GCN', 'GAT', or 'MPNN'.")
        self.fc = Linear(hidden_dim if model_type != 'MPNN' else 2 * hidden_dim, 1)
        self.relu = ReLU()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.model_type in ['GCN', 'GAT']:
            x = self.conv1(x, edge_index)
            x = self.relu(x)
            x = self.conv2(x, edge_index)
            x = self.relu(x)
            x = global_mean_pool(x, batch)
        elif self.model_type == 'MPNN':
            for _ in range(3):
                x = self.conv(x, edge_index)
                x = self.relu(x)
            x = self.set2set(x, batch)
        x = self.fc(x)
        return x.squeeze(-1)