import copy
from typing import List, Optional

import torch
import torch.nn.functional as F
import torch_geometric.nn.pool
from torch.nn import ModuleList
from torch_geometric.nn import GCNConv
from torchvision.ops import MLP
from torch_geometric.nn import GCN
from torch_geometric.data import Data


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


class CustomPygGCN(torch.nn.Module):
    """Wrapper for torch geometric GCN, to allow for OrbitIndivGCN forward() method"""
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, out_channels: int):
        super().__init__()
        self.gcn = GCN(in_channels, hidden_channels, num_layers, out_channels)

    def init_conv(self, in_channels, out_channels: int, **kwargs):
        return self.gcn.init_conv(in_channels, out_channels, **kwargs)

    def reset_parameters(self):
        self.gcn.reset_parameters()

    def forward(
            self,
            x: torch.tensor,
            edge_index: torch.tensor,
            orbits: Optional[List[torch.tensor]],
    ) -> torch.tensor:
        return self.gcn(x, edge_index)

    @torch.no_grad()
    def inference(self, loader, device=None, progress_bar=False):
        return self.gcn.inference(loader, device, progress_bar)

    def __repr__(self):
        return self.gcn.__repr__()


class RniGCN(CustomPygGCN):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, out_channels: int, rni_channels: int):
        super().__init__(in_channels + rni_channels, hidden_channels, num_layers, out_channels)
        self.rni_channels = rni_channels

    def forward(self, x, edge_index, orbits):
        # x: [batch * num_nodes, in_channels]
        noise = torch.rand(x.size()[0], self.rni_channels)  # [batch * num_nodes, in_channels]
        extended_x = torch.cat((x, noise), dim=1)  # [batch * num_nodes, in_channels + rni_channels]
        return super().forward(extended_x, edge_index, orbits)


class RniMaxPoolGCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, out_channels: int, rni_channels: int):
        super().__init__()
        assert num_layers > 0
        in_channels = in_channels + rni_channels
        self.rni_channels = rni_channels
        self.out_channels = out_channels

        self.conv_layers = []
        self.mlp_merge_layers = []  # used to combine output of GCN and max pooling
        for i in range(num_layers):
            input_size = in_channels if i == 0 else hidden_channels
            output_size = out_channels if i == num_layers - 1 else hidden_channels
            self.conv_layers.append(GCNConv(input_size, hidden_channels))  # conv always goes to hidden channels
            self.mlp_merge_layers.append(MLP(in_channels=hidden_channels * 2, hidden_channels=[output_size]))
        self.conv_layers = ModuleList(self.conv_layers)
        self.mlp_merge_layers = ModuleList(self.mlp_merge_layers)

    def forward(self, x, edge_index, orbits):
        # x: [batch * num_nodes, in_channels]
        noise = torch.rand(x.size()[0], self.rni_channels)  # [batch * num_nodes, in_channels]
        x = torch.cat((x, noise), dim=1)  # [batch * num_nodes, in_channels + rni_channels]

        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index)  # [batch * num_nodes, hidden_channels]
            max_pool = torch_geometric.nn.pool.global_max_pool(x, batch=None)  # [1, hidden_channels]
            max_pool = max_pool.expand(x.size()[0], x.size()[1])  # [batch * num_nodes, hidden_channels]
            combined_x = torch.cat((x, max_pool), dim=1)  # [batch * num_nodes, 2 * hidden_channels]
            x = self.mlp_merge_layers[i](combined_x)

            # do not use activation function in the final layer
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
        return x


# does not use one-hot encodings, since graphs have varying sizes
class UniqueIdGCN(CustomPygGCN):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, out_channels: int):
        super().__init__(in_channels + 1, hidden_channels, num_layers, out_channels)

    def forward(self, x, edge_index, orbits):
        # x: [batch * num_nodes, in_channels]
        ids = torch.unsqueeze(torch.arange(1, x.size()[0] + 1), dim=1)  # [batch * num_nodes, 1]
        extended_x = torch.cat((x, ids), dim=1)  # [batch * num_nodes, in_channels + 1]
        return super().forward(extended_x, edge_index, orbits)


# does not use one-hot encodings, since graphs have varying sizes
class OrbitIndivGCN(CustomPygGCN):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, out_channels: int):
        super().__init__(in_channels, hidden_channels, num_layers, out_channels=hidden_channels)
        # MLP with single hidden layer
        self.mlp = MLP(in_channels=hidden_channels + 1, hidden_channels=[hidden_channels] + [out_channels])

    def forward(self, x, edge_index, orbits):
        # x: [batch * num_nodes, in_channels]
        gcn_output = super().forward(x, edge_index, orbits)  # [batch * num_nodes, hidden_channels]
        # just range for now, no one-hot
        unique_ids = torch.arange(0, x.size()[0])  # [batch * num_nodes]
        ids = torch.empty_like(unique_ids)  # [batch * num_nodes]
        for orbit in orbits:
            # compute unique IDs for each orbit
            # e.g. orbit = [0, 1, 4, 6] means that unique_ids_to_append = [0, 1, -, -, 2, -, 3, ...]
            ids[orbit] = unique_ids[0:orbit.size()[0]]

        # append per-orbit unique IDs to GCN output
        ids = torch.unsqueeze(ids, dim=1)  # [batch * num_nodes, 1]
        extended_gcn_output = torch.cat((gcn_output, ids), dim=1)  # [batch * num_nodes, hidden_channels + 1]

        # pass through final MLP
        return self.mlp(extended_gcn_output)


class MaxOrbitGCN(CustomPygGCN):
    def __init__(self, in_channels: int, hidden_channels: int, num_layers: int, out_channels: int, max_orbit: int):
        super().__init__(in_channels, hidden_channels, num_layers, out_channels=hidden_channels)
        self.max_orbit = max_orbit
        mlp_output_size = max_orbit * out_channels
        # MLP with single hidden layer to convert to proper output size
        self.mlp = MLP(in_channels=hidden_channels, hidden_channels=[hidden_channels] + [mlp_output_size])

    def forward(self, x, edge_index, orbits):
        # x: [batch * num_nodes, in_channels]
        gcn_output = super().forward(x, edge_index, orbits)  # [batch * num_nodes, out_channels]
        mlp_output = self.mlp(gcn_output)  # [batch * num_nodes, max_orbit * out_channels]

        # reshape tensor to match flattened target, so that it can be compared using cross entropy
        # [batch * num_nodes * max_orbit, out_channels]
        mlp_output = torch.reshape(mlp_output, (mlp_output.size()[0] * self.max_orbit, -1))
        return mlp_output
