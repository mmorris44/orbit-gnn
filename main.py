import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import from_networkx, to_networkx
import torch_geometric.datasets as torch_datasets
import numpy as np
from typing import List, Optional
import random


def get_orbits(graph: nx.Graph):
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


def plot_labeled_graph(graph: nx.Graph, orbits: Optional[List[List[int]]] = None, show_node_id: bool = False):
    pos = nx.spring_layout(graph, seed=1)
    options = {"edgecolors": "tab:gray", "node_size": 800, "alpha": 1}

    node_color = [0] * len(graph.nodes)
    if orbits is not None:
        node_color = [0] * len(graph.nodes)
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

mutag_dataset = torch_datasets.TUDataset(root='./datasets', name='MUTAG')
mutag_nx = []
for graph in mutag_dataset:
    graph_nx = to_networkx(graph, to_undirected=True, remove_self_loops=True, node_attrs=['x'])
    # convert node attributes from one-hot encoding into number
    node_attributes = nx.get_node_attributes(graph_nx, 'x')
    for node, attribute in node_attributes.items():
        attribute = np.array(attribute)
        non_zero_index = np.nonzero(attribute)[0][0]
        print(non_zero_index)
        node_attributes[node] = non_zero_index
    nx.set_node_attributes(graph_nx, node_attributes, 'x')
    mutag_nx.append(graph_nx)
random.shuffle(mutag_nx)  # shuffle dataset

print('--- MUTAG orbits ---')
for graph in mutag_nx:
    orbits = get_orbits(graph)
    print(get_orbits(graph))
    plot_labeled_graph(graph, orbits=orbits)
