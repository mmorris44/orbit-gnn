import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import from_networkx
import torch_geometric.datasets as torch_datasets


def get_orbits(graph: nx.Graph):
    node_list = list(graph.nodes)[1:]
    orbits = [[list(graph.nodes)[0]]]
    isomorphisms = [iso for iso in nx.vf2pp_all_isomorphisms(G, G, node_label='x')]

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


G = nx.Graph()

G.add_nodes_from([
    (0, {'x': (1.0, 1.0), 'y': 1}),
    (1, {'x': (2.0, 1.0), 'y': 1}),
])

for i in range(2, 7):
    G.add_node(i, **{'x': (1.0, 1.0), 'y': 1})

G.add_edges_from([
    (0, 1), (1, 2), (1, 3), (3, 4), (4, 5), (4, 6)
])

print(G.nodes)
print(G.edges)
print('orbits:', get_orbits(graph=G))
print()

# Convert the graph into PyTorch geometric
pyg_graph = from_networkx(G)

print('graph:\n', pyg_graph, '\n\n')
print('x:\n', pyg_graph.x, '\n\n')
print('y:\n', pyg_graph.y, '\n\n')
print('edge index:\n', pyg_graph.edge_index, '\n\n')

mutag_dataset = torch_datasets.TUDataset(root='./datasets', name='MUTAG')
print(mutag_dataset)
