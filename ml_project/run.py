import logging
import os
import time

import hydra
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import OmegaConf

from src.config.config import Config
from src.constants.consts import APP_NAME
from src.constants.enums import EProgramMode
from src.data.make_dataset import prepare_data
from src.models.train_model import main_train_model

logger = logging.getLogger(APP_NAME)


@hydra.main(
    config_path="configs",
    config_name="config",
)
def my_app(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg))
    temp = OmegaConf.structured(Config)
    working_dir = get_original_cwd()
    print(f"Orig working directory    : {working_dir}")
    print(f"Current working directory : {os.getcwd()}")
    if EProgramMode.PrepareData in cfg.mode:
        logger.info(f"STAGE: PrepareData")
        prepare_data(
            cfg.prepare_data,
            to_absolute_path(cfg.train_dataset),
            to_absolute_path(cfg.test_dataset),
            to_absolute_path(cfg.vocab_path),
        )
    if EProgramMode.Train in cfg.mode:
        logger.info(f"STAGE: Train")
        main_train_model(cfg)
    if EProgramMode.Predict in cfg.mode:
        # TODO: implement prediction
        pass


if __name__ == "__main__":
    # try: python run.py +train/model=rnn
    my_app()
