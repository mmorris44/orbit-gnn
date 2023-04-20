from collections import namedtuple
from typing import List, Tuple, Optional

import torch.nn
from torch_geometric.data import Data

from datasets import MaxOrbitGCNTransform


def model_accuracy(
        dataset: List[Data],
        model: torch.nn.Module,
        device: str,
        max_orbit_transform: Optional[MaxOrbitGCNTransform] = None,
) -> Tuple[float, float, float]:
    """Compute orbit-equivariant model accuracy on the given dataset.

    For each graph, node-level, orbit-level, and graph-level accuracy are computed.
    These are then averaged over all the graphs.
    Function assumes that class labels are given in data.y (not one-hot).

    :param dataset: Dataset to compute accuracy on.
    :param model: GNN model to compute accuracy of.
    :param device: torch device to be used.
    :param max_orbit_transform: if using a max-orbit GCN transformation, use to convert model output for evaluation
    :return: Tuple[node accuracy, orbit accuracy, graph accuracy]
    """
    # track 3 accuracies
    total_node_accuracy = 0
    total_orbit_accuracy = 0
    total_graph_accuracy = 0

    for data in dataset:
        # compute predictions and ground truth
        data = data.to(device)
        out = model(data.x, data.edge_index, orbits=data.orbits)

        # transform model output if using a max-orbit transform
        if max_orbit_transform is not None:
            out = max_orbit_transform.transform_output(out, data)

        predictions = torch.argmax(out, dim=1)  # no need to softmax, since it's monotonic
        ground_truth = data.y  # assume class labels are given in data.y

        nodes_correct = 0
        orbits_correct = 0

        # check correctness of each orbit
        orbits = data.orbits
        for orbit in orbits:
            orbit_predictions = predictions[orbit].tolist()
            orbit_ground_truth = ground_truth[orbit].tolist()
            # compute size of multiset intersection between predictions and ground truth
            intersection_count = 0
            for prediction in orbit_predictions:
                if prediction in orbit_ground_truth:
                    orbit_ground_truth.remove(prediction)
                    intersection_count += 1

            # update node and orbit counts
            nodes_correct += intersection_count
            if intersection_count == len(orbit):
                orbits_correct += 1

        graphs_correct = 1 if orbits_correct == len(data.orbits) else 0

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
                torch.tensor([0, 1, 2, 3, 4]),
            ]),
        CustomData(
            x=torch.tensor([
                [9, 5, 1, 6],
                [8, 3, 0, 12],
                [5, 0, 13, 0],
                [10, 0, 34, 0],
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
