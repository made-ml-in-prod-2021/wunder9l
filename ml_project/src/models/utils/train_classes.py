from abc import ABCMeta, abstractmethod
from collections import Callable
from typing import Optional

import numpy as np
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

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
        """Common train one preocedure (run model->get predicts->call loss function)"""
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


def make_one_batch_runner(
    runner_type: EOneBatchType, model: nn.Module, loss_fn: Callable, is_train: bool
):
    if runner_type == EOneBatchType.Common:
        return CommonOneBatchRunner(model, loss_fn, is_train)
    else:
        raise NotImplementedError(f"No suitable IOneBatchRunner for {runner_type}")


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
    one_batch_runner: EOneBatchType,
    model: nn.Module,
    loss_fn: Callable,
    train_dataloader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    is_train: bool,
):
    train_one_batch = make_one_batch_runner(one_batch_runner, model, loss_fn, is_train)
    train_one_epoch = TrainOneEpoch(
        train_dataloader, train_one_batch, optimizer, device, is_train
    )
    return train_one_epoch
