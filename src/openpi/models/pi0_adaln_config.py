import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
from typing_extensions import override

from openpi.models import pi0_config as _pi0_config
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    from openpi.models.pi0_adaln import Pi0AdaLN


@dataclasses.dataclass(frozen=True)
class Pi0AdaLNConfig(_pi0_config.Pi0Config):
    """Forked Pi0 temporal model that injects memory via adaptive layer norm instead of cross-attention."""

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0AdaLN":
        from openpi.models.pi0_adaln import Pi0AdaLN

        return Pi0AdaLN(self, rngs=nnx.Rngs(rng))
