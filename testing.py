from collections import namedtuple
from typing import List, Tuple

import torch.nn
from torch_geometric.data import Data


def model_accuracy(dataset: List[Data], model: torch.nn.Module, device: str) -> Tuple[float, float, float]:
    """Compute equivariant model accuracy on the given dataset.

    :param dataset: Dataset to compute accuracy on.
    :param model: GNN model to compute accuracy of.
    :param device: torch device to be used.
    :return: Tuple[node accuracy, orbit accuracy, graph accuracy]
    """
    total_node_accuracy = 0
    total_orbit_accuracy = 0
    total_graph_accuracy = 0

    for data in dataset:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        predictions = torch.argmax(out, dim=1)  # no need to softmax, since it's monotonic
        ground_truth = data.y  # assume class labels are given in data.y

        nodes_correct = 0
        orbits_correct = 0

        orbits = data.orbits
        for orbit in orbits:
            orbit_predictions = predictions[orbit].tolist()
            orbit_ground_truth = ground_truth[orbit].tolist()
            print('orbit_predictions', orbit_predictions)
            print('orbit_ground_truth', orbit_ground_truth)
            intersection_count = 0
            for prediction in orbit_predictions:
                if prediction in orbit_ground_truth:
                    orbit_ground_truth.remove(prediction)
                    intersection_count += 1

            nodes_correct += intersection_count
            if intersection_count == len(orbit):
                orbits_correct += 1

        graphs_correct = 1 if orbits_correct == len(data.orbits) else 0

        print(nodes_correct, orbits_correct, graphs_correct)

        total_node_accuracy += nodes_correct / predictions.size()[0]
        total_orbit_accuracy += orbits_correct / len(orbits)
        total_graph_accuracy += graphs_correct

    return total_node_accuracy / len(dataset), total_orbit_accuracy / len(dataset), total_graph_accuracy / len(dataset)


def test_model_accuracy():
    model = lambda x, y: x

    class CustomData:
        x: torch.tensor
        y: torch.tensor
        orbits: List[torch.tensor]
        edge_index: torch.tensor

        def __init__(self, x, y, orbits):
            self.x = x
            self.y = y
            self.orbits = orbits
            self.edge_index = torch.zeros(1)

        def to(self, device: str):
            return self

    dataset = [
        CustomData(
            x=torch.tensor([
                [9, 5, 1, 6],
                [8, 3, 0, 12],
                [5, 0, 13, 0],
                [34, 0, 10, 0],
                [0, 10, 0, 2],
            ]),
            y=torch.tensor([3, 0, 2, 2, 1]),
            orbits=[
                torch.tensor([0, 1, 3]),
                torch.tensor([2, 4]),
            ]),
    ]
    node_acc, orbit_acc, graph_acc = model_accuracy(dataset, model, 'cpu')
    print(node_acc, orbit_acc, graph_acc)


if __name__ == '__main__':
    test_model_accuracy()
