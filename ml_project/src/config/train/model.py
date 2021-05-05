from dataclasses import dataclass
from typing import Any
from omegaconf import MISSING

from src.constants.enums import EModelType


@dataclass
class ModelRNNArgs:
    input_size: int = 100
    hidden_size: int = 200
    num_layers: int = 1
    dropout: bool = True
    bidirectional: bool = False


@dataclass
class ModelArgs:
    model_args: Any = MISSING
    model_type: EModelType = EModelType.RNN
