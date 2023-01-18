import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, num_node_features: int, num_classes: int, gcn_layers=2, hidden_size=16):
        super().__init__()
        assert gcn_layers > 0
        self.conv_layers = []
        for i in range(gcn_layers):
            input_size = num_node_features if i == 0 else hidden_size
            output_size = num_classes if i == gcn_layers - 1 else hidden_size
            self.conv_layers.append(GCNConv(input_size, output_size))
        self.conv_layers = ModuleList(self.conv_layers)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, conv in enumerate(self.conv_layers):
            # final layer
            if i == len(self.conv_layers) - 1:
                x = conv(x, edge_index)
                return x

            # any other layer
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)


class RniGCN(GCN):
    def __init__(self, num_node_features: int, num_classes: int, gcn_layers=2, hidden_size=16, noise_dims=2):
        super().__init__(num_node_features + noise_dims, num_classes, gcn_layers, hidden_size)
        self.noise_dims = noise_dims

    def forward(self, data):
        # data.x: [batch * num_nodes, num_node_features]
        noise = torch.rand(data.x.shape[0], self.noise_dims)  # [batch * num_nodes, noise_dims]
        data.x = torch.cat((data.x, noise), dim=1)  # [batch * num_nodes, num_node_features + noise_dims]
        return super().forward(data)
