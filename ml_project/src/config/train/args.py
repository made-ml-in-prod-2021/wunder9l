from dataclasses import dataclass, field
from typing import List, Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from src.config.train.model import ModelArgs, ModelRNNArgs
from src.constants.enums import (
    ELossType,
    EOneBatchType,
    EOptimizerType,
    ELRSchedulerType,
    EModelType,
)


@dataclass
class LRSchedulerArgs:
    type: ELRSchedulerType = ELRSchedulerType.No
    step_size: int = 10
    gamma: float = 0.1


defaults = [
    {"train/model": "rnn"},
]


def default_model_args() -> ModelArgs:
    return ModelArgs(model_type=EModelType.RNN, model_args=ModelRNNArgs())


@dataclass
class TrainArgs:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    gpu: bool = True
    test_size: float = 0.2
    epochs: int = 40
    batch_size: int = 64
    model: ModelArgs = default_model_args()
    loss_fn: ELossType = ELossType.EBCEWithLogitsLoss
    optimizer: EOptimizerType = EOptimizerType.Adam
    learning_rate: float = 1e-1
    dump_model: str = "models/model.dmp"
    lr_scheduler: LRSchedulerArgs = LRSchedulerArgs()
    interactive: bool = False
    report_path: str = "reports/train_report.csv"


def register_train_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="train/model",
        name="rnn",
        node=ModelArgs(model_type=EModelType.RNN, model_args=ModelRNNArgs()),
    )
