import copy
from typing import List

import torch_geometric.datasets as torch_datasets
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import to_networkx, from_networkx
import networkx as nx

from wl import get_wl_orbits


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


# each returned graph will contain node attributes 'y' for target outputs
def get_orbit_molecule_dataset(dataset: List[nx.Graph]) -> List[nx.Graph]:
    orbit_dataset = []
    trivial_orbits_only_count = 0
    for graph_index, graph in enumerate(dataset):
        _, orbits = get_wl_orbits(graph)  # maybe change this in future to use actual orbits?
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
        target_node_index = chosen_orbit[graph_index % len(chosen_orbit)]

        current_node_attributes = nx.get_node_attributes(graph, 'x')
        node_attributes = {node: {'x': current_node_attributes[node], 'y': 1 if node == target_node_index else 0}
                           for node in graph.nodes}

        orbit_graph = copy.deepcopy(graph)
        nx.set_node_attributes(orbit_graph, node_attributes)
        orbit_dataset.append(orbit_graph)

    # print('Number of graphs with only trivial orbits:', trivial_orbits_only_count, '//', len(dataset))
    return orbit_dataset


def get_pyg_dataloader_from_nx(nx_graphs: List[nx.Graph], batch_size=8) -> DataLoader:
    pyg_list = []
    for graph in nx_graphs:
        pyg_list.append(from_networkx(graph))
    return DataLoader(pyg_list, batch_size=batch_size)
