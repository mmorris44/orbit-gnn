from typing import Tuple, List

import networkx as nx

from graph_theory import compute_orbits
from plotting import plot_labeled_graph


# input is a list of ints but is treated as unordered by sorting first
def multi_set_hash_function(input_set: list) -> int:
    return hash(tuple(sorted(input_set)))


def wl(graph: nx.Graph) -> Tuple[int, List[int]]:
    labels = [-1] * len(graph)
    node_attributes = nx.get_node_attributes(graph, 'x')
    if node_attributes:
        for node in graph.nodes:
            labels[node] = hash(node_attributes[node])  # initial labels are hashes
    num_unique = len(set(labels))

    for wl_iteration in range(1, len(graph) + 1):
        previous_labels = labels[:]
        previous_num_unique = num_unique
        global_hash = multi_set_hash_function(previous_labels)

        for node in graph.nodes:
            neighbours = graph[node]
            neighbour_labels = []
            for neighbour in neighbours:
                neighbour_labels.append(previous_labels[neighbour])
            neighbour_hash = multi_set_hash_function(neighbour_labels)
            combined_hash = hash((previous_labels[node], neighbour_hash, global_hash))
            labels[node] = combined_hash

        num_unique = len(set(labels))
        if num_unique == previous_num_unique:
            # orbit WL has converged
            return wl_iteration, labels
    raise Exception('WL did not converge: something is wrong with the algorithm')


def compute_wl_hash(graph: nx.Graph) -> int:
    _, final_labels = wl(graph)
    return multi_set_hash_function(final_labels)


def compute_wl_orbits(graph: nx.Graph) -> Tuple[int, List[List[int]]]:
    n_iterations, final_labels = wl(graph)
    node_list = list(graph.nodes)[1:]
    orbits = [[list(graph.nodes)[0]]]

    for node in node_list:
        found_orbit = False
        for orbit_index, orbit in enumerate(orbits):
            orbit_node = orbit[0]

            if final_labels[node] == final_labels[orbit_node]:
                orbits[orbit_index].append(node)
                found_orbit = True
        if not found_orbit:
            orbits.append([node])

    return n_iterations, orbits


def check_orbits_against_wl(
        nx_dataset: List[nx.Graph],
        max_graph_size_to_check: int = 1000,
        plot_counter_examples=True,
):
    orbit_size_counts = {i: 0 for i in range(1, 1001)}  # track how many of each orbit size there is
    skip_count = 0
    for i, graph in enumerate(nx_dataset):
        # print(i, '//', len(enzymes_nx), '| #nodes =', len(graph), '| #edges =', graph.number_of_edges())
        if len(graph) > max_graph_size_to_check:
            # print('graph too large to check for now, skipping')
            skip_count += 1
            continue
        orbits = compute_orbits(graph)
        for orbit in orbits:
            orbit_size_counts[len(orbit)] += 1
        orbit_wl_iterations, wl_orbits = compute_wl_orbits(graph)
        if not orbits == wl_orbits:
            print('counter-example found:')
            print('orbits:   ', orbits)
            print('WL orbits:', wl_orbits)
            print('orbit-WL iterations:', orbit_wl_iterations)
            if plot_counter_examples:
                plot_labeled_graph(graph, orbits=orbits)
    print('done checking:', skip_count, 'graphs skipped')
    print('orbit sizes:', {size: orbit_size_counts[size] for size in range(1, 1001) if orbit_size_counts[size] > 0})
