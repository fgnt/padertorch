from pathlib import Path
import typing as tp

from torch import Tensor


TPath = tp.Union[str, Path]
TSeqLen = tp.Optional[tp.List[int]]
TActivationFn = tp.Union[str, tp.Callable]
TSeqReturn = tp.Tuple[Tensor, TSeqLen]
TDevice = tp.Union[str, int, tp.Sequence[int]]
