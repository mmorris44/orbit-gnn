import copy

import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn import GCNConv
from torchvision.ops import MLP
from torch_geometric.nn import GCN


class DeprecatedCustomGCN(torch.nn.Module):
    def __init__(self, num_node_features: int, num_classes: int, gcn_layers=2, hidden_size=64):
        super().__init__()
        assert gcn_layers > 0
        self.num_classes = num_classes
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
            # x = F.dropout(x, training=self.training)


class MlpGCN(DeprecatedCustomGCN):
    def __init__(self, num_node_features: int, num_classes: int, gcn_layers=2, hidden_size=16,
                 max_graph_size=100, mlp_hidden_size=(16,)):
        super().__init__(num_node_features, num_classes, gcn_layers, hidden_size)
        input_size = num_node_features if gcn_layers <= 1 else hidden_size  # input size if only one layer
        self.conv_layers[-1] = GCNConv(input_size, hidden_size)  # correct size of final GCN layer

        # Node outputs will be stacked together with 0s to pad smaller graphs
        self.mlp = MLP(hidden_size * max_graph_size, list(mlp_hidden_size) + [num_classes * max_graph_size])

    def forward(self, data):
        output = super().forward(data)
        # TODO: stack, pad, and pass to MLP
        return output


class UniqueIdDeepSetsGCN(DeprecatedCustomGCN):
    def __init__(self, num_node_features: int, num_classes: int, gcn_layers=2, hidden_size=16,
                 theta_mlp_sizes=(16, 20), rho_mlp_sizes=(16,)):
        super().__init__(num_node_features, num_classes, gcn_layers, hidden_size)
        input_size = num_node_features if gcn_layers <= 1 else hidden_size  # input size if only one layer
        self.conv_layers[-1] = GCNConv(input_size, hidden_size)  # correct size of final GCN layer

        self.theta_mlp = MLP(1 + hidden_size, theta_mlp_sizes)  # inner function of DeepSets
        self.rho_mlp = MLP(theta_mlp_sizes[-1], list(rho_mlp_sizes) + [num_classes])  # outer function

    def forward(self, data):
        # currently will only work for batch size of 1
        gcn_output = super().forward(data)  # [batch * num_nodes, hidden_size]
        final_outputs = torch.zeros((gcn_output.shape[0], self.num_classes))  # [batch * num_nodes, num_classes]
        for node in range(gcn_output.shape[0]):  # for each node
            unique_id_tensor = torch.zeros((gcn_output.shape[0], 1))  # [batch * num_nodes, 1]
            unique_id_tensor[node, 0] = node

            inner_input = torch.cat((unique_id_tensor, gcn_output), dim=1)  # [batch * num_nodes, 1 + hidden_size]
            inner_output = self.theta_mlp(inner_input)  # [batch * num_nodes, theta_mlp_sizes[-1]]

            summed_outer_input = torch.sum(inner_output, dim=0)  # [theta_mlp_sizes[-1]]
            outer_output = self.rho_mlp(summed_outer_input)  # [num_classes]
            final_outputs[node] = outer_output
        return final_outputs


class RniGCN(GCN):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, out_channels: int, rni_channels: int):
        super().__init__(in_channels + rni_channels, hidden_channels, num_layers, out_channels)
        self.rni_channels = rni_channels

    def forward(self, x, edge_index, **kwargs):
        # x: [batch * num_nodes, in_channels]
        noise = torch.rand(x.size()[0], self.rni_channels)  # [batch * num_nodes, in_channels]
        extended_x = torch.cat((x, noise), dim=1)  # [batch * num_nodes, in_channels + rni_channels]
        return super().forward(extended_x, edge_index)


# does not use one-hot encodings, since graphs have varying sizes
class UniqueIdGCN(GCN):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, out_channels: int):
        super().__init__(in_channels + 1, hidden_channels, num_layers, out_channels)

    def forward(self, x, edge_index, **kwargs):
        # x: [batch * num_nodes, in_channels]
        ids = torch.unsqueeze(torch.range(1, x.size()[0]), dim=1)  # [batch * num_nodes, 1]
        extended_x = torch.cat((x, ids), dim=1)  # [batch * num_nodes, in_channels + 1]
        return super().forward(extended_x, edge_index)
