import random
import networkx as nx
from torch_geometric.utils.convert import from_networkx

from graph_theory import get_orbits
from wl import check_orbits_against_wl
from datasets import get_nx_molecule_dataset


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
print(nx.get_node_attributes(G, 'x'))
print(nx.get_node_attributes(G, 'y'))
print('orbits:', get_orbits(graph=G))
print()

# Convert the graph into PyTorch geometric
pyg_graph = from_networkx(G)

print('graph:\n', pyg_graph, '\n\n')
print('x:\n', pyg_graph.x, '\n\n')
print('y:\n', pyg_graph.y, '\n\n')
print('edge index:\n', pyg_graph.edge_index, '\n\n')

mutag_nx = get_nx_molecule_dataset('MUTAG')
random.shuffle(mutag_nx)  # shuffle dataset

print('\n--- MUTAG orbits ---')
check_orbits_against_wl(mutag_nx)

enzymes_nx = get_nx_molecule_dataset('ENZYMES')
# random.shuffle(enzymes_nx)

print('\n--- ENZYMES orbits ---')
check_orbits_against_wl(enzymes_nx, max_graph_size_to_check=66)  # can do 66 in <30 seconds

proteins_nx = get_nx_molecule_dataset('PROTEINS')
# random.shuffle(enzymes_nx)

print('\n--- PROTEINS orbits ---')
check_orbits_against_wl(proteins_nx, max_graph_size_to_check=60)  # can do 60 in <10 seconds
