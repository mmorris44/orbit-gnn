from typing import Optional

import torch
from torch import Tensor
from torch.nn.modules.loss import _WeightedLoss
from torch.nn import functional as F


class OrbitSortingCrossEntropyLoss(_WeightedLoss):
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        reordered_target = self._reordered_target(input, target)
        return F.cross_entropy(input, reordered_target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction,
                               label_smoothing=self.label_smoothing)

    @staticmethod
    def _reordered_target(input: Tensor, target: Tensor) -> Tensor:
        # 1. Make input and target categorical
        input_categorical = torch.argmax(input, dim=1)
        target_categorical = torch.argmax(target, dim=1)

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


# loss = OrbitSortingCrossEntropyLoss()
# input = torch.tensor([
#     [9, 5, 1, 6],
#     [8, 3, 0, 12],
#     [5, 0, 13, 0],
#     [34, 0, 10, 0],
#     [0, 10, 0, 2],
# ])
# print('input\n', input)
# target = torch.tensor([
#     [0, 0, 0, 1],
#     [0, 0, 1, 0],
#     [1, 0, 0, 0],
#     [0, 0, 1, 0],
#     [0, 1, 0, 0],
# ])
# print('target\n', target)
# reordered_target = loss._reordered_target(input, target)
# print('reordered_target\n', reordered_target)
