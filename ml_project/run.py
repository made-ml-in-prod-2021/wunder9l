import hydra
from omegaconf import OmegaConf

from src.config.config import Config


@hydra.main(
    config_path="configs",
    config_name="config",
)
def my_app(cfg: Config) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    # try: python run.py +train/model=rnn
    my_app()
