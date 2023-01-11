from typing import Tuple, List

import networkx as nx


# input is a list of ints but is treated as unordered by sorting first
def multi_set_hash_function(input_set: list) -> int:
    return hash(tuple(sorted(input_set)))


def orbit_wl(graph: nx.Graph) -> Tuple[int, List[int]]:
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


def get_wl_orbits(graph: nx.Graph) -> Tuple[int, List[List[int]]]:
    n_iterations, final_labels = orbit_wl(graph)
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
