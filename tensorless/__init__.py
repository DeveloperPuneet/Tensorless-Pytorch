"""
Tensorless PyTorch
==========

ML with maximum automation and minimum setup.

    import tensorless as tl

    tl.train("./data")
    tl.run("model.tl")

See https://github.com/DeveloperPuneet/Tensorless-Pytorch for full documentation.
"""

from ._version import __version__
from .api import inspect, load, pretrain, run, train
from .config import TrainConfig
from .errors import (
    CheckpointError,
    ConfigError,
    DataError,
    ModelError,
    SerializationError,
    TensorlessError,
)

__all__ = [
    "train",
    "pretrain",
    "run",
    "load",
    "inspect",
    "TrainConfig",
    "TensorlessError",
    "DataError",
    "ConfigError",
    "ModelError",
    "CheckpointError",
    "SerializationError",
    "__version__",
]
