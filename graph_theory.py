from typing import List

import networkx as nx


def get_orbits(graph: nx.Graph) -> List[List[int]]:
    node_list = list(graph.nodes)[1:]
    orbits = [[list(graph.nodes)[0]]]
    isomorphisms = [iso for iso in nx.vf2pp_all_isomorphisms(graph, graph, node_label='x')]

    for node in node_list:
        found_orbit = False
        for orbit_index, orbit in enumerate(orbits):
            orbit_node = orbit[0]
            for isomorphism in isomorphisms:
                if isomorphism[node] == orbit_node or isomorphism[orbit_node] == node:
                    found_orbit = True
                    break
            if found_orbit:
                orbits[orbit_index].append(node)
                break
        if not found_orbit:
            orbits.append([node])
    return orbits
