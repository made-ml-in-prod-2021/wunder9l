from dataclasses import dataclass, field
from typing import List

from hydra.core.config_store import ConfigStore

from src.config.train.args import TrainArgs, register_train_config
from src.constants.enums import EProgramMode

default_program_modes = [
    EProgramMode.PrepareData,
    EProgramMode.Train,
    EProgramMode.Predict,
]


@dataclass
class PrepareDataArgs:
    input_file: str = "data/raw/spam.csv"


@dataclass
class Config:
    mode: List[EProgramMode] = field(default_factory=lambda: default_program_modes)
    train: TrainArgs = TrainArgs()
    prepare_data: PrepareDataArgs = PrepareDataArgs()


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
register_train_config()
