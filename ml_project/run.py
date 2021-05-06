import logging
import time

import hydra
from omegaconf import OmegaConf

from src.config.config import Config
from src.constants.enums import EProgramMode
from src.data.make_dataset import prepare_data
from src.models.train_model import main_train_model

logger = logging.getLogger(__file__)


@hydra.main(
    config_path="configs",
    config_name="config",
)
def my_app(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg))
    if EProgramMode.PrepareData in cfg.mode:
        logger.info(f"STAGE: PrepareData")
        start = time.time()
        prepare_data(cfg.prepare_data.input_file, cfg.train.dataset_filename)
        logger.info(f"STAGE: PrepareData finished in {time.time() - start}")
    if EProgramMode.Train in cfg.mode:
        logger.info(f"STAGE: Train")
        start = time.time()
        main_train_model(cfg.train)
        logger.info(f"STAGE: Train finished in {time.time() - start}")
    if EProgramMode.Predict in cfg.mode:
        # TODO: implement prediction
        pass


if __name__ == "__main__":
    # try: python run.py +train/model=rnn
    my_app()
