from abc import ABCMeta, abstractmethod
from collections import Callable
from typing import Optional

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
        return loss


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
        self.model.train(self.is_train)

        batch_size, max_length = data.shape
        hid_state = torch.zeros((batch_size, self.model.hidden_state))
        logprobs = []

        for x_t in data.transpose(0, 1):
            hid_state, logp_next = self.model(
                x_t, hid_state
            )  # <-- here we call your one-step code
            logprobs.append(logp_next)

        return torch.stack(logprobs, dim=1)


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


class TrainOneEpoch(object):
    def __init__(
        self,
        dataloader: DataLoader,
        one_batch_train: IOneBatchRunner,
        optimizer: Optional[torch.optim.Optimizer],
        device: torch.device,
        is_train: bool,
    ):
        self.dataloader = dataloader
        self.one_batch_train = one_batch_train
        self.optimizer = optimizer
        self.device = device
        self.is_train = is_train

    def __call__(self) -> np.ndarray:
        loss_hist = []
        for batch in tqdm(
            self.dataloader, total=len(self.dataloader), desc="training..."
        ):
            data = batch[DATA].to(self.device)
            target = batch[TARGET].to(self.device)

            loss = self.one_batch_train(data, target)
            loss_hist.append(loss.item())

            if self.optimizer and self.is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return np.mean(loss_hist)


def make_one_epoch_runner(
    model_args: ModelArgs,
    model: nn.Module,
    loss_fn: Callable,
    train_dataloader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    is_train: bool,
):
    train_one_batch = make_one_batch_runner(
        model_args, model, loss_fn, is_train
    )
    train_one_epoch = TrainOneEpoch(
        train_dataloader, train_one_batch, optimizer, device, is_train
    )
    return train_one_epoch
