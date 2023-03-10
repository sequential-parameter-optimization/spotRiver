"""Datasets.

This module contains a collection of datasets for multiple tasks: classification, regression, etc.
The data corresponds to popular datasets and are conveniently wrapped to easily iterate over
the data in a stream fashion. All datasets have fixed size.

"""
from . import base, synth
from .airline_passengers import AirlinePassengers


__all__ = [
    "AirlinePassengers",
    "base",
    "synth",
]
