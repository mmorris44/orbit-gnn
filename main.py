import random
import networkx as nx

from wl import check_orbits_against_wl
from datasets import nx_molecule_dataset, orbit_molecule_dataset, pyg_dataset_from_nx

# G = nx.Graph()
#
# G.add_nodes_from([
#     (0, {'x': (1.0, 1.0), 'y': 1}),
#     (1, {'x': (2.0, 1.0), 'y': 1}),
# ])
#
# for i in range(2, 7):
#     G.add_node(i, **{'x': (1.0, 1.0), 'y': 1})
#
# G.add_edges_from([
#     (0, 1), (1, 2), (1, 3), (3, 4), (4, 5), (4, 6)
# ])

mutag_nx = nx_molecule_dataset('MUTAG')
# random.shuffle(mutag_nx)  # shuffle dataset
enzymes_nx = nx_molecule_dataset('ENZYMES')
proteins_nx = nx_molecule_dataset('PROTEINS')

orbit_mutag_nx = orbit_molecule_dataset(mutag_nx, num_features=7)
orbit_mutag_dataset = pyg_dataset_from_nx(orbit_mutag_nx)
pyg_graph = orbit_mutag_dataset[0]
print('graph:\n', pyg_graph, '\n\n')
print('x:\n', pyg_graph.x, '\n\n')
print('y:\n', pyg_graph.y, '\n\n')
print('edge index:\n', pyg_graph.edge_index, '\n\n')


# print('\n--- MUTAG orbits ---')
# check_orbits_against_wl(mutag_nx)

# print('\n--- ENZYMES orbits ---')
# check_orbits_against_wl(enzymes_nx, max_graph_size_to_check=66)  # can do 66 in <30 seconds

# print('\n--- PROTEINS orbits ---')
# check_orbits_against_wl(proteins_nx, max_graph_size_to_check=60)  # can do 60 in <10 seconds
