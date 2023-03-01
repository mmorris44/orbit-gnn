import copy
from typing import List

import torch_geometric.datasets as torch_datasets
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx, from_networkx
import networkx as nx

from wl import compute_wl_orbits


def nx_molecule_dataset(name='MUTAG') -> List[nx.Graph]:
    torch_dataset = torch_datasets.TUDataset(root='./datasets', name=name)
    return nx_from_torch_dataset(torch_dataset)


def nx_from_torch_dataset(torch_dataset: List[Data]) -> List[nx.Graph]:
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


# each returned graph will contain node attributes 'y' for target outputs
# one-hot encode the 'x' values
# see https://pytorch-geometric.readthedocs.io/en/latest/notes/data_cheatsheet.html for #features
def orbit_molecule_dataset(dataset: List[nx.Graph], num_features: int) -> List[nx.Graph]:
    orbit_dataset = []
    trivial_orbits_only_count = 0
    for graph_index, graph in enumerate(dataset):
        _, orbits = compute_wl_orbits(graph)  # maybe change this in future to use actual orbits?
        # find the first >=2-sized orbit
        non_trivial_orbit_index = -1
        for i, orbit in enumerate(orbits):
            if len(orbit) > 1:
                non_trivial_orbit_index = i
                break
        if non_trivial_orbit_index == -1:  # just pick one of the trivial orbits
            trivial_orbits_only_count += 1
            non_trivial_orbit_index = graph_index % len(orbits)
        # assign node from orbit as target
        chosen_orbit = orbits[non_trivial_orbit_index]

        # one-hot encode the node attributes
        current_node_attributes = nx.get_node_attributes(graph, 'x')
        for node, attribute in current_node_attributes.items():
            one_hot_encoding = [0.0] * num_features
            one_hot_encoding[attribute] = 1.0
            current_node_attributes[node] = tuple(one_hot_encoding)

        # y for node i will be e.g. [0, 0, 1, 0],
        # where len(y) = len(chosen_orbit),
        # y[j] == 1 only once,
        # and only if (i in chosen_orbit)

        node_attributes = {node: {'x': current_node_attributes[node],
                                  'y': tuple([1 if node == target_node_index else 0
                                              for target_node_index in chosen_orbit])}
                           for node in graph.nodes}

        orbit_graph = copy.deepcopy(graph)
        nx.set_node_attributes(orbit_graph, node_attributes)
        orbit_dataset.append(orbit_graph)

    # print('Number of graphs with only trivial orbits:', trivial_orbits_only_count, '//', len(dataset))
    return orbit_dataset


# will one-hot encode the attributes
def pyg_dataset_from_nx(nx_graphs: List[nx.Graph]) -> List[Data]:
    pyg_list = []
    for graph in nx_graphs:
        pyg_list.append(from_networkx(graph))
    return pyg_list
