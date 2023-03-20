import copy
from typing import List

import torch
import torch_geometric.datasets as torch_datasets
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx, from_networkx
import networkx as nx
import networkx.algorithms.isomorphism as iso

import plotting
from graph_theory import compute_orbits
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


# combine the input and output graphs by adding y-values to the input graphs
# target graph will differ from input graph by one node (or none)
def combined_bioisostere_dataset(
        inputs: List[Data],
        targets: List[Data],
        no_change_input_option=True,
        only_equivariant=False,
):
    """Combine bioisostere inputs and targets into one dataset.

    :param inputs: source molecules
    :param targets: optimal bioisosteres
    :param no_change_input_option: for each node, add element to vector which when 1 implies no change from input
    :param only_equivariant: filter out non-equivariant examples from dataset
    :return:
    """
    combined_graphs: List[Data] = []

    # nx versions of graphs
    nx_inputs = nx_from_torch_dataset(inputs)
    nx_targets = nx_from_torch_dataset(targets)

    num_requiring_symmetry_breaking = 0  # keep track of how many swapped nodes come from orbits with >1 node
    num_bioisosteres = 0  # keep track of how many have target different from input

    node_match_fn = iso.numerical_node_match('x', 0)  # function to check for node equality

    for graph_index in range(len(inputs)):
        input_graph, target_graph = nx_inputs[graph_index], nx_targets[graph_index]

        # check if already isomorphic
        input_target_same = nx.is_isomorphic(input_graph, target_graph, node_match=node_match_fn)
        if not input_target_same:
            num_bioisosteres += 1

        # for each node of input graph, set the node label to -1 in both input and target graph
        # graphs are not necessarily in the same order, so need to have an inner loop for target graph
        # then check for isomorphism between those two graphs
        # if graphs are isomorphic, then this is the node that was swapped out

        swapped_out_node = -1  # which node was swapped out
        swap_out_for = -1  # what was it swapped out for

        for input_swap_node in input_graph.nodes:
            # construct input graph to compare
            input_node_attributes = nx.get_node_attributes(input_graph, 'x')  # get current node attributes
            input_node_attributes[input_swap_node] = -1  # update swap node attribute
            input_graph_compare = copy.deepcopy(input_graph)
            input_node_attributes = {node: {'x': input_node_attributes[node]}
                                     for node in input_graph_compare.nodes}
            nx.set_node_attributes(input_graph_compare, input_node_attributes)  # set attributes of compare graph

            for target_swap_node in target_graph.nodes:
                # construct target graph to compare, same as above, but use target_swap_node
                target_node_attributes = nx.get_node_attributes(target_graph, 'x')
                target_node_attributes[target_swap_node] = -1
                target_graph_compare = copy.deepcopy(target_graph)
                target_node_attributes = {node: {'x': target_node_attributes[node]}
                                          for node in target_graph_compare.nodes}
                nx.set_node_attributes(target_graph_compare, target_node_attributes)

                # check for isomorphism between input and target compare graphs
                if nx.is_isomorphic(input_graph_compare, target_graph_compare, node_match=node_match_fn):
                    swapped_out_node = input_swap_node
                    swap_out_for = nx.get_node_attributes(target_graph, 'x')[target_swap_node]
                    break

            if swapped_out_node != -1:  # isomorphism found in inner loop
                break
        if swapped_out_node == -1:
            # plotting.plot_labeled_graph(input_graph, input_graph_orbits)
            # plotting.plot_labeled_graph(target_graph, compute_orbits(target_graph))
            # issues created by mmpdb not copying floating atoms to the bioisostere
            # must be manually fixed in the dataset csv file
            print('Issue with graph', graph_index)
            raise Exception('Could not compute which node swaps to form the bioisostere')

        # get graph orbits to save to features
        input_graph_orbits = compute_orbits(input_graph)

        # check if task requires symmetry breaking
        graph_requires_symmetry_breaking = False
        for orbit in input_graph_orbits:
            if swapped_out_node in orbit:  # find the orbit the node is from
                if len(orbit) > 1 and not input_target_same:
                    # if bioisostere (target diff from input) and orbit bigger than 1
                    num_requiring_symmetry_breaking += 1
                    graph_requires_symmetry_breaking = True
                break

        # possibly skip graph that requires symmetry breaking
        if only_equivariant and graph_requires_symmetry_breaking:
            continue

        # append combined graph
        input_graph_pyg = inputs[graph_index]
        if no_change_input_option:
            # last element =1 means no change from input
            combined_y = torch.zeros(input_graph_pyg.x.size()[0], input_graph_pyg.x.size()[1] + 1)
        else:
            combined_y = torch.zeros_like(input_graph_pyg.x)

        for node in input_graph.nodes:
            if node == swapped_out_node and not input_target_same:
                combined_y[node, swap_out_for] = 1.0  # one-hot of new swapped node value
            elif no_change_input_option:
                combined_y[node, -1] = 1.0  # set flag for no change of node value
            else:
                combined_y[node, :] = input_graph_pyg.x[node]  # do not change node value
        combined_graph = Data(x=input_graph_pyg.x, edge_index=input_graph_pyg.edge_index,
                              edge_attr=input_graph_pyg.edge_attr, y=combined_y, orbits=input_graph_orbits)
        combined_graphs.append(combined_graph)

    print('--- Constructed combined bioisostere dataset ---')
    print('Actual bioisosteres:', num_bioisosteres, '/', len(inputs))
    print('Bioisosteres requiring symmetry breaking:', num_requiring_symmetry_breaking, '/', num_bioisosteres)
    return combined_graphs


