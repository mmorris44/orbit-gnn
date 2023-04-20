from typing import Optional, List

import torch
from torch import Tensor
from torch.nn.modules.loss import _WeightedLoss
from torch.nn import functional as F


class CrossEntropyLossWrapper(_WeightedLoss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super().__init__(weight, size_average, reduce, reduction)
        self.cross_entropy_loss\
            = torch.nn.CrossEntropyLoss(weight, size_average, ignore_index, reduce, reduction, label_smoothing)

    def forward(self, input: Tensor, target: Tensor, non_equivariant_orbits: List[Tensor]) -> Tensor:
        return self.cross_entropy_loss(input, target)


class OrbitSortingCrossEntropyLoss(_WeightedLoss):
    """
    Compute loss after re-ordering the targets within each non-equivariant orbit.
    Re-ordering is done to match the permutation corresponding to sorting the prediction and ground truth, then pairing.
    The loss assumes that target is categorical, and not one-hot.
    """
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, input: Tensor, target: Tensor, non_equivariant_orbits: List[Tensor]) -> Tensor:
        """Compute orbit loss by aligning sorted input and target within non-equivariant orbits.
        Passing the full list of orbits to the function also works, but wastes computation.

        :param input: value for each class (highest = chosen class) - [n_nodes, n_classes]
        :param target: class indices for each node - [n_nodes]
        :param non_equivariant_orbits: indices of nodes in each non-equivariant orbit - each is [n_nodes_in_orbit]
        :return: loss
        """
        reordered_target = OrbitSortingCrossEntropyLoss._reorder_all_orbits(input, target, non_equivariant_orbits)
        return F.cross_entropy(input, reordered_target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction,
                               label_smoothing=self.label_smoothing)

    @staticmethod
    def _reorder_all_orbits(input: Tensor, target: Tensor, non_equivariant_orbits: List[Tensor]) -> Tensor:
        # collect nodes from each orbit that is non-equivariant
        # compute a re-ordered target for each collection
        # combine re-ordered targets together and compute a full cross-entropy

        reordered_target = target
        for orbit in non_equivariant_orbits:
            # orbit is a tensor with shape [n_nodes_in_orbit]
            orbit_input = input[orbit]
            orbit_target = target[orbit]
            reordered_orbit_target = OrbitSortingCrossEntropyLoss._reordered_target(orbit_input, orbit_target)
            reordered_target[orbit] = reordered_orbit_target
        return reordered_target

    @staticmethod
    def _reordered_target(input: Tensor, target: Tensor) -> Tensor:
        # 1. Make input and target categorical
        input_categorical = torch.argmax(input, dim=1)
        target_categorical = target  # target is already categorical

        # 2. Sort input and target
        _, input_sorted_indices = torch.sort(input_categorical)
        _, target_sorted_indices = torch.sort(target_categorical)

        # 3. Pair up input and target, compute original target permutation that results in same pairing
        # input_sorted_indices[1] is the index of the second-lowest value in input_categorical
        identity_permutation = torch.arange(0, input_categorical.size()[0], dtype=torch.long)
        inverse_input_sorted_permutation = torch.empty_like(identity_permutation)
        inverse_input_sorted_permutation[input_sorted_indices] = identity_permutation  # invert permutation
        target_reorder_permutation = target_sorted_indices[inverse_input_sorted_permutation]  # compose

        # 4. Re-order target with permutation
        reordered_target = target[target_reorder_permutation]
        return reordered_target


def test_reorder_all_orbits():
    loss = OrbitSortingCrossEntropyLoss()
    input = torch.tensor([
        [9, 5, 1, 6],
        [8, 3, 0, 12],
        [5, 0, 13, 0],
        [34, 0, 10, 0],
        [0, 10, 0, 2],
    ])
    input_categorical = torch.argmax(input, dim=1)
    print('input_categorical\n', input_categorical)
    class_target = torch.tensor([3, 0, 2, 2, 1])
    print('target\n', class_target)
    orbits = [
        torch.tensor([0, 1, 3]),
        torch.tensor([2, 4]),
    ]
    print('orbits\n', orbits)
    reordered_target = loss._reorder_all_orbits(input, class_target, orbits)
    print('reordered_target\n', reordered_target)


def test_reordered_target():
    loss = OrbitSortingCrossEntropyLoss()
    input = torch.tensor([
        [9, 5, 1, 6],
        [8, 3, 0, 12],
        [5, 0, 13, 0],
        [34, 0, 10, 0],
        [0, 10, 0, 2],
    ])
    print('input\n', input)
    probability_target = torch.tensor([
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
    ])
    class_target = torch.tensor([3, 2, 0, 2, 1])
    print('target\n', class_target)
    reordered_target = loss._reordered_target(input, class_target)
    print('reordered_target\n', reordered_target)


if __name__ == '__main__':
    test_reorder_all_orbits()
