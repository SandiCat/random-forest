from dataclasses import dataclass, field
from typing import *
from enum import Enum, auto


class Model(Enum):
    ID3 = auto()
    RF = auto()


@dataclass
class Config:
    mode: str
    model: Model
    max_depth: Optional[int]
    num_trees: int
    feature_ratio: float
    example_ratio: float


def lookup_with_default(dict, key, default):
    if key in dict:
        return dict[key]
    else:
        return default


def load_config(filename: str) -> Config:
    with open(filename) as f:
        raw_values = dict(tuple(line.split("=")) for line in f.read().splitlines())
        max_depth_label = "max_depth"
        max_depth = None
        if max_depth_label in raw_values:
            max_depth_parsed = int(raw_values[max_depth_label])
            if max_depth_parsed != -1:
                max_depth = max_depth_parsed

        return Config(
            mode=raw_values["mode"],
            model=Model[raw_values["model"]],
            max_depth=max_depth,
            num_trees=int(lookup_with_default(raw_values, "num_trees", "1")),
            example_ratio=float(lookup_with_default(raw_values, "example_ratio", "1")),
            feature_ratio=float(lookup_with_default(raw_values, "feature_ratio", "1")),
        )
