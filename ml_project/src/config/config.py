from dataclasses import dataclass, field
from typing import List, Optional

from hydra.core.config_store import ConfigStore

from src.config.train.args import TrainArgs, register_train_config
from src.constants.enums import EProgramMode

default_program_modes = [
    EProgramMode.PrepareData,
    EProgramMode.Train,
    EProgramMode.Predict,
    EProgramMode.Visualize
]


@dataclass
class PrepareDataArgs:
    train_file: str = "data/raw/spam.csv"
    test_file: str = "data/raw/spam.csv"
    pretrained_vectors: Optional[str] = None
    vectors_cache_directory: str = ".vector_cache"
    tokenizer_name: str = "basic_english"


@dataclass
class PredictArgs:
    result_file: str = "reports/prediction.csv"


@dataclass
class Config:
    mode: List[EProgramMode] = field(default_factory=lambda: default_program_modes)
    train: TrainArgs = TrainArgs()
    prepare_data: PrepareDataArgs = PrepareDataArgs()
    predict: PredictArgs = PredictArgs()
    train_dataset: str = "data/processed/spam.csv"
    test_dataset: str = "data/processed/spam.csv"
    vocab_path: str = "data/processed/vocab.pkl"


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(name="train_only", node=Config)
register_train_config()
