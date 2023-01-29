"""Synthetic datasets.

Each synthetic dataset is a stream generator. The benefit of using a generator is that they do not
store the data and each data sample is generated on the fly. Except for a couple of methods,
the majority of these methods are infinite data generators.

"""
from .sea import SEA

__all__ = [
    "SEA",
]
