from typing import List

import torch_geometric.datasets as torch_datasets
import numpy as np
from torch_geometric.utils.convert import to_networkx
import networkx as nx


def get_nx_molecule_dataset(name='MUTAG') -> List[nx.Graph]:
    torch_dataset = torch_datasets.TUDataset(root='./datasets', name=name)
    nx_dataset = []
    for graph in torch_dataset:
        graph_nx = to_networkx(graph, to_undirected=True, remove_self_loops=True, node_attrs=['x'])
        # convert node attributes from one-hot encoding into number
        node_attributes = nx.get_node_attributes(graph_nx, 'x')
        for node, attribute in node_attributes.items():
            attribute = np.array(attribute)
            non_zero_index = np.nonzero(attribute)[0][0]
            node_attributes[node] = non_zero_index
        nx.set_node_attributes(graph_nx, node_attributes, 'x')
        nx_dataset.append(graph_nx)
    return nx_dataset
