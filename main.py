import random
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import from_networkx
from typing import List, Optional

from graph_theory import get_orbits
from wl import get_wl_orbits
from datasets import get_nx_molecule_dataset


def plot_labeled_graph(graph: nx.Graph, orbits: Optional[List[List[int]]] = None, show_node_id: bool = True):
    pos = nx.spring_layout(graph, seed=1)
    options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 1}

    node_color = [0] * len(graph)
    if orbits is not None:
        node_color = [0] * len(graph)
        for node in graph.nodes:
            orbit_index = 0
            for i, orbit in enumerate(orbits):
                if node in orbit:
                    orbit_index = i
                    break
            node_color[node] = orbit_index + 1
            if len(orbits[orbit_index]) == 1:
                node_color[node] = 0  # do not color nodes that are in their own orbit

    nx.draw_networkx_nodes(graph, pos, **options, node_color=node_color, cmap=plt.cm.tab20)
    nx.draw_networkx_edges(graph, pos, width=1, alpha=0.5)
    labels = nx.get_node_attributes(graph, 'x')
    # append node ID to label
    if show_node_id:
        for node, label in labels.items():
            labels[node] = str(node) + ':' + str(label)
    nx.draw_networkx_labels(graph, pos, labels, font_size=10, font_color='black')
    plt.tight_layout()
    plt.axis("off")
    plt.show()


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

print('--- MUTAG orbits ---')
for graph in mutag_nx:
    orbits = get_orbits(graph)
    orbit_WL_iterations, WL_orbits = get_wl_orbits(graph)
    if not orbits == WL_orbits:
        print('orbits:   ', orbits)
        print('WL orbits:', WL_orbits)
        print('orbit-WL iterations:', orbit_WL_iterations)
        plot_labeled_graph(graph, orbits=orbits)

enzymes_nx = get_nx_molecule_dataset('ENZYMES')
# random.shuffle(enzymes_nx)

print('--- ENZYMES orbits ---')
skip_count = 0
for i, graph in enumerate(enzymes_nx):
    # print(i, '//', len(enzymes_nx), '| #nodes =', len(graph), '| #edges =', graph.number_of_edges())
    if len(graph) > 30:  # can do 66 in <20 seconds
        # print('graph too large to check for now, skipping')
        skip_count += 1
        continue
    orbits = get_orbits(graph)
    orbit_WL_iterations, WL_orbits = get_wl_orbits(graph)
    if not orbits == WL_orbits:
        print('orbits:   ', orbits)
        print('WL orbits:', WL_orbits)
        print('orbit-WL iterations:', orbit_WL_iterations)
        plot_labeled_graph(graph, orbits=orbits)
print('done checking:', skip_count, 'graphs skipped')

proteins_nx = get_nx_molecule_dataset('PROTEINS')
# random.shuffle(enzymes_nx)

print('--- PROTEINS orbits ---')
skip_count = 0
for i, graph in enumerate(proteins_nx):
    if len(graph) > 60:  # can do 60 in <10 seconds
        skip_count += 1
        continue
    orbits = get_orbits(graph)
    orbit_WL_iterations, WL_orbits = get_wl_orbits(graph)
    if not orbits == WL_orbits:
        print('orbits:   ', orbits)
        print('WL orbits:', WL_orbits)
        print('orbit-WL iterations:', orbit_WL_iterations)
        plot_labeled_graph(graph, orbits=orbits)
print('done checking:', skip_count, 'graphs skipped')
