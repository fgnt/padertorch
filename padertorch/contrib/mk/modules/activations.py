from torch import Tensor
from torch.nn import GELU as _GELU


class GELU(_GELU):
    """Magnitude-preserving GELU activation function."""
    scale: float = 0.653

    def __init__(
        self, approximate: str = 'none', magnitude_preserving: bool = False
    ):
        super().__init__(approximate=approximate)
        self.magnitude_preserving = magnitude_preserving

    def forward(self, input: Tensor) -> Tensor:
        output = super().forward(input)
        if self.magnitude_preserving:
            return output / self.scale
        return output
