import json
from . import base


class RiverHyperDict(base.FileConfig):
    """River hyperparameter dictionary."""

    def __init__(self):
        super().__init__(
            filename="river_hyper_dict.json",
        )

    def load(self):
        with open(self.path, "r") as f:
            d = json.load(f)
        return d
