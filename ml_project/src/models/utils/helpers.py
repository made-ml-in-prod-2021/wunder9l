from typing import List

import pandas as pd
import torch
from torch import nn as nn

from src.config.train.args import LRSchedulerArgs
from src.config.train.model import ModelRNNArgs, ModelArgs
from src.constants.enums import ELossType, EModelType, EOptimizerType, ELRSchedulerType


def plot_train_val_loss(train_loss_hist: List[float], val_loss_hist: List[float]):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"For plotting need to install matplotlib")
        return
    x = list(range(len(train_loss_hist)))
    plt.plot(x, train_loss_hist, label="train")
    plt.plot(x, val_loss_hist, label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def save_train_report(
    train_score_hist: List[float], val_score_hist: List[float], path: str
):
    df = pd.DataFrame(
        dict(
            epoch=range(1, len(train_score_hist) + 1),
            train_loss=train_score_hist,
            val_loss=val_score_hist,
        )
    )
    df.to_csv(path)


def make_rnn_model(args: ModelRNNArgs) -> nn.Module:
    rnn = nn.RNN(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    )
    return rnn


def make_loss_fn(loss_type: ELossType):
    MAP = {
        ELossType.ENLLLoss: nn.NLLLoss,
    }
    return MAP[loss_type]()


def make_optimizer(
    optimizer_type: EOptimizerType, model: nn.Module, learning_rate: float
):
    MAP = {EOptimizerType.Adam: torch.optim.Adam}
    return MAP[optimizer_type](model.parameters(), lr=learning_rate)


def get_device(gpu: bool) -> torch.device:
    if gpu and torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def make_model(args: ModelArgs):
    if args.model_type == EModelType.RNN:
        return make_rnn_model(args.model_args)
    else:
        raise NotImplementedError(
            f"Required model {args.model_type} is not implemented"
        )


def make_lr_scheduler(optimizer: torch.optim.Optimizer, args: LRSchedulerArgs):
    if args.type == ELRSchedulerType.Step:
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )
    else:
        return None
