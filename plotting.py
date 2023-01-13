import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Optional


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
