from dataclasses import dataclass
from omegaconf import MISSING

from src.config.train.model import ModelArgs
from src.constants.enums import (
    ELossType,
    EOneBatchType,
    EOptimizerType,
    ELRSchedulerType,
)


@dataclass
class LRSchedulerArgs:
    type: ELRSchedulerType = ELRSchedulerType.No
    step_size: int = 10
    gamma: float = 0.1


@dataclass
class TrainArgs:
    gpu: bool = True
    dataset_filename: str = "data/processed/data.csv"
    test_size: float = 0.2
    tokenizer_name: str = "basic_english"
    pretrained_vectors: str = "glove.6B.100d"
    epochs: int = 40
    batch_size: int = 64
    model: ModelArgs = MISSING
    loss_fn: ELossType = MISSING
    optimizer: EOptimizerType = EOptimizerType.Adam
    learning_rate: float = 1e-1
    one_batch_runner: EOneBatchType = EOneBatchType.Common
    dump_model: str = "models/model.dmp"
    lr_scheduler: LRSchedulerArgs = MISSING
    interactive: bool = False
    report_path: str = "models/train_report.csv"
