from abc import ABCMeta, abstractmethod
from collections import Callable
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.train.model import ModelArgs
from src.constants.consts import TARGET, DATA
from src.constants.enums import EOneBatchType


class IOneBatchRunner:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, data: torch.Tensor, target: torch.Tensor):
        """Return loss from one batch

        Args:
            data - input of model (batch_size x input_data)

            target - output of model (batch_size x output_data)
        """


class CommonOneBatchRunner(IOneBatchRunner):
    def __init__(self, model: nn.Module, loss_fn: Callable, is_train: bool):
        """Common train one batch procedure (run model->get predicts->call loss function)"""
        self.model = model
        self.loss_fn = loss_fn
        self.is_train = is_train

    def __call__(self, data: torch.Tensor, target: torch.Tensor):
        """Return loss from one batch

        Args:
            data - input of model (batch_size x input_data)

            target - output of model (batch_size x output_data)
        """
        self.model.train(self.is_train)
        if self.is_train:
            predictions = self.model(data).cpu()
        else:
            with torch.no_grad():
                predictions = self.model(data).cpu()
        loss = self.loss_fn(predictions, target)
        return loss, predictions


class RnnOneBatchRunner(IOneBatchRunner):
    def __init__(
        self,
        model: nn.Module,
        loss_fn: Callable,
        min_sequence_to_loss: int,
        is_train: bool,
    ):
        """Rnn-specific train one batch procedure.

        Run model->get predicts and hidden state->call next step

        Args:
            model - rnn model

            loss_fn - loss function

            min_sequence_to_loss - minimal length of sequence to calculate loss function
            (for train only). To make learning faster

            is_train - training or validation
        """
        self.model = model
        self.loss_fn = loss_fn
        self.min_sequence_to_loss = min_sequence_to_loss
        self.is_train = is_train

    def __call__(self, data: torch.Tensor, target: torch.Tensor):
        """Return loss from one batch. Run on every token from batch.

         For example, if there are 10 tokens per sample in batch then it
         would run for first token (with generated h0 as hidden state),
         get it's output and hidden state h1, after that call it for second
         token (with h1 as hidden state), get h2 etc...

        Args:
            data - input of model (batch_size x input_data)

            target - output of model (batch_size x output_data)
        """
        seq_len, batch_size = data.shape
        self.model.train(self.is_train)
        if self.is_train:
            predictions, hidden_state = self.model(data, None)
        else:
            with torch.no_grad():
                predictions, hidden_state = self.model(data, None)
        min_sequence_to_loss = min(self.min_sequence_to_loss, seq_len - 1)
        to_loss = predictions[min_sequence_to_loss:].cpu()
        target = (
            target.repeat(to_loss.shape[0], 1)
            .reshape(to_loss.shape)
            .to(dtype=to_loss.dtype)
        )
        loss = self.loss_fn(to_loss, target)
        return loss, predictions[-1].detach().cpu()


def make_one_batch_runner(
    model_args: ModelArgs, model: nn.Module, loss_fn: Callable, is_train: bool
):
    if model_args.one_batch_runner == EOneBatchType.Common:
        return CommonOneBatchRunner(model, loss_fn, is_train)
    elif model_args.one_batch_runner == EOneBatchType.Rnn:
        return RnnOneBatchRunner(
            model,
            loss_fn,
            min_sequence_to_loss=model_args.model_args.min_sequence_to_loss,
            is_train=is_train,
        )
    else:
        raise NotImplementedError(f"No suitable IOneBatchRunner for {model_args}")


class OneEpochRunner(object):
    def __init__(
        self,
        dataloader: DataLoader,
        one_batch_train: IOneBatchRunner,
        optimizer: Optional[torch.optim.Optimizer],
        device: torch.device,
        is_train: bool,
        score_functions: Optional[
            Dict[str, Callable[[torch.Tensor, torch.Tensor], float]]
        ],
    ):
        self.dataloader = dataloader
        self.one_batch_train = one_batch_train
        self.optimizer = optimizer
        self.device = device
        self.is_train = is_train
        self.score_functions = score_functions

    def __call__(self) -> Tuple[np.ndarray, Dict[str, float]]:
        loss_hist = []
        target_hist, prediction_hist = [], []
        desc = "training..." if self.is_train else "validating..."
        for batch in tqdm(self.dataloader, total=len(self.dataloader), desc=desc):
            data = batch[DATA].to(self.device)
            target = batch[TARGET].to(self.device)

            loss, batch_predictions = self.one_batch_train(data, target)
            loss_hist.append(loss.item())
            if self.score_functions:
                target_hist.append(target)
                prediction_hist.append(batch_predictions)

            if self.optimizer and self.is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        score = self._call_score_functions(target_hist, prediction_hist)
        return np.mean(loss_hist), score

    def _call_score_functions(
        self, target_hist: List[torch.Tensor], prediction_hist: List[torch.Tensor]
    ) -> Dict[str, float]:
        if not self.score_functions:
            return {}
        target_hist = torch.cat(target_hist, dim=0)
        prediction_hist = torch.cat(prediction_hist, dim=0)
        score = {
            name: fn(target_hist, prediction_hist)
            for name, fn in self.score_functions.items()
        }
        return score


def make_one_epoch_runner(
    model_args: ModelArgs,
    model: nn.Module,
    loss_fn: Callable,
    train_dataloader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    is_train: bool,
    score_functions: Optional[
        Dict[str, Callable[[torch.Tensor, torch.Tensor], float]]
    ] = None,
):
    train_one_batch = make_one_batch_runner(model_args, model, loss_fn, is_train)
    train_one_epoch = OneEpochRunner(
        train_dataloader, train_one_batch, optimizer, device, is_train, score_functions
    )
    return train_one_epoch
