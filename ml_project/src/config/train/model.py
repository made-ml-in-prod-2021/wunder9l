from dataclasses import dataclass
from typing import Any, Optional
from omegaconf import MISSING

from src.constants.enums import EModelType, EOneBatchType


@dataclass
class ModelRNNArgs:
    pretrained_vectors: Optional[str] = None
    input_size: int = 100
    hidden_size: int = 200
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = False
    num_classes: int = 1
    min_sequence_to_loss: int = 30


@dataclass
class ModelArgs:
    model_args: Any = MISSING
    model_type: EModelType = EModelType.RNN
    one_batch_runner: EOneBatchType = EOneBatchType.Rnn
