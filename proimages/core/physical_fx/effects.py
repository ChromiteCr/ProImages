from collections.abc import Callable

import numpy as np

from proimages.core.physical_fx.grain import apply_grain
from proimages.core.physical_fx.halation import apply_halation
from proimages.core.physical_fx.rolloff import apply_rolloff
from proimages.core.physical_fx.vignette import apply_vignette

EffectFn = Callable[[np.ndarray], np.ndarray]

STAGES: list[EffectFn] = [
    apply_grain,
    apply_halation,
    apply_rolloff,
    apply_vignette,
]
