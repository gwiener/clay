"""Layout engine registry and lookup."""

from clay.layout import LayoutEngine
from clay.layout.random import Random
from clay.layout.energy import Energy
from clay.layout.ranked import Ranked

ENGINES: dict[str, type[LayoutEngine]] = {
    "random": Random,
    "energy": Energy,
    "ranked": Ranked,
}


def get_engine(name: str) -> type[LayoutEngine]:
    """Get engine class by name."""
    if name not in ENGINES:
        raise ValueError(f"Unknown layout engine: {name}. Available: {list(ENGINES.keys())}")
    return ENGINES[name]
