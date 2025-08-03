from typing import Sequence, Union

import torch

Number = Union[int, float]
TensorLike = Union[torch.Tensor, Sequence[Number], Sequence[Sequence[Number]]]
