import logging
from subprocess import Popen
from typing import List, Dict

import pandas as pd
import torch
from hydra.utils import to_absolute_path
from sklearn.metrics import accuracy_score, recall_score, precision_score
from torch import nn as nn
from torchtext.vocab import Vocab

from src.config.train.args import LRSchedulerArgs
from src.config.train.model import ModelArgs
from src.constants.consts import PAD, APP_NAME
from src.constants.enums import ELossType, EModelType, EOptimizerType, ELRSchedulerType
from src.models.adapted_models.rnn import make_rnn_model


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


def save_train_report(results: Dict[str, List[float]], path: str):
    results["epoch"] = list(range(1, max(len(items) for items in results.values()) + 1))
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)


def run_visualization(original_dataset: str, train_result: str, model_path: str) -> None:
    cmd = [
        "streamlit",
        "run",
        to_absolute_path("src/visualization/visualize.py"),
        "--",
        "--original_dataset",
        original_dataset,
        "--train_result",
        train_result,
        "--model",
        model_path,
    ]
    logging.getLogger(APP_NAME).info(f"Visualization, running cmd: {cmd}")
    Popen(cmd).wait()


def make_loss_fn(loss_type: ELossType):
    MAP = {
        ELossType.ENLLLoss: nn.NLLLoss,
        ELossType.EBCEWithLogitsLoss: nn.BCEWithLogitsLoss,
    }
    return MAP[loss_type]()


def make_optimizer(
    optimizer_type: EOptimizerType, model: nn.Module, learning_rate: float
):
    MAP = {
        EOptimizerType.Adam: torch.optim.Adam,
        EOptimizerType.SGD: torch.optim.SGD,
    }
    return MAP[optimizer_type](model.parameters(), lr=learning_rate)


def get_device(gpu: bool) -> torch.device:
    if gpu and torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def make_model(args: ModelArgs, vocab: Vocab) -> nn.Module:
    if args.model_type == EModelType.RNN:
        return make_rnn_model(vocab.vectors, vocab[PAD], args.model_args)
    else:
        raise NotImplementedError(
            f"Required model {args.model_type} is not implemented"
        )


def load_model(args: ModelArgs, vocab: Vocab, dump_file: str) -> nn.Module:
    model = make_model(args, vocab)
    with open(dump_file, "rb") as fp:
        state_dict = torch.load(fp)
        model.load_state_dict(state_dict)
    return model


def make_lr_scheduler(optimizer: torch.optim.Optimizer, args: LRSchedulerArgs):
    if args.type == ELRSchedulerType.Step:
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.step_size, gamma=args.gamma
        )
    else:
        return None


def from_raw_to_labels(y_pred: torch.Tensor):
    """Returns labels (1.0 or 0.0) from raw regression values (-inf, inf)"""
    return torch.round(torch.sigmoid(y_pred))


def accuracy_from_predictions(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_pred_rounded = from_raw_to_labels(y_pred)
    score = accuracy_score(y_true, y_pred_rounded)
    return score


def precision_from_predictions(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_pred_rounded = from_raw_to_labels(y_pred)
    score = precision_score(y_true, y_pred_rounded)
    return score


def recall_from_predictions(y_true: torch.Tensor, y_pred: torch.Tensor):
    y_pred_rounded = from_raw_to_labels(y_pred)
    score = recall_score(y_true, y_pred_rounded)
    return score
