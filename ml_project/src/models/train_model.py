import logging
from collections import Callable
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.config.train.args import TrainArgs
from src.constants.consts import PAD
from src.data.make_dataset import read_datasets

from src.models.utils.helpers import (
    make_loss_fn,
    get_device,
    make_model,
    make_optimizer,
    make_lr_scheduler,
    plot_train_val_loss,
    save_train_report,
)
from src.models.utils.train_classes import (
    TrainOneEpoch,
    make_one_epoch_runner,
)
from src.utils.dataset_utils import make_text_dataloader
from src.utils.decorators import time_it

logger = logging.getLogger(__file__)


def train_cycle(
    model: nn.Module,
    train_one_epoch: TrainOneEpoch,
    validate_one_epoch: TrainOneEpoch,
    epochs: int,
    dump_model_filename: str,
    lr_scheduler,
    plot_fn: Optional[Callable[List[float], List[float], None]],
) -> Tuple[List[float], List[float]]:
    best_val_loss = np.inf
    train_loss_hist = []
    val_loss_hist = []
    for epoch in range(epochs):
        train_loss = train_one_epoch()
        val_loss = validate_one_epoch()
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        message = (
            f"Epoch #{epoch}: train loss: {train_loss:.5f}, val loss: {val_loss:.5f}"
        )
        logger.info(message)
        if lr_scheduler:
            lr_scheduler.step()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with open(dump_model_filename, "wb") as fp:
                torch.save(model.state_dict(), fp)
        if plot_fn:
            plot_fn(train_loss, val_loss_hist)

    return train_loss_hist, val_loss_hist


@time_it("main_train_model, duration", logger.info)
def main_train_model(args: TrainArgs):
    train_dataset, val_dataset, vocab = read_datasets(
        args.dataset_filename,
        args.test_size,
        args.tokenizer_name,
        args.pretrained_vectors,
    )
    train_dataloader = make_text_dataloader(
        train_dataset, args.batch_size, train_dataset.vocab[PAD]
    )
    val_dataloader = make_text_dataloader(
        train_dataset, args.batch_size, train_dataset.vocab[PAD]
    )

    model = make_model(args.model)
    loss_fn = make_loss_fn(args.loss_fn)

    optimizer = make_optimizer(args.optimizer, model, args.learning_rate)
    device = get_device(args.gpu)

    train_one_epoch = make_one_epoch_runner(
        args.one_batch_runner,
        model,
        loss_fn,
        train_dataloader,
        optimizer,
        device,
        is_train=True,
    )
    val_one_epoch = make_one_epoch_runner(
        args.one_batch_runner,
        model,
        loss_fn,
        val_dataloader,
        None,
        device,
        is_train=False,
    )

    lr_scheduler = make_lr_scheduler(optimizer, args.lr_scheduler)
    plot_fn = plot_train_val_loss if args.interactive else None
    train_score_hist, val_score_hist = train_cycle(
        model,
        train_one_epoch,
        val_one_epoch,
        args.epochs,
        args.dump_model,
        lr_scheduler,
        plot_fn,
    )
    save_train_report(train_score_hist, val_score_hist, args.report_path)
