import logging
from collections import Callable
from typing import Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn
from hydra.utils import to_absolute_path

from src.config.config import Config
from src.constants.consts import PAD, APP_NAME
from src.data.make_dataset import load_processed_dataset
from src.models.utils.helpers import (
    make_loss_fn,
    get_device,
    make_model,
    make_optimizer,
    make_lr_scheduler,
    plot_train_val_loss,
    save_train_report,
    accuracy_from_predictions,
    recall_from_predictions,
    precision_from_predictions,
    run_visualization,
)
from src.models.utils.train_classes import (
    OneEpochRunner,
    make_one_epoch_runner,
)
from src.utils.dataset_utils import make_text_dataloader, ensure_path
from src.utils.decorators import time_it

logger = logging.getLogger(APP_NAME)


def to_column_view(results: List[Dict[str, float]], prefix) -> Dict[str, List[float]]:
    def get_column(key):
        return [item[key] for item in results]

    keys = {k for item in results for k in item}
    out = {prefix + key: get_column(key) for key in keys}
    return out


def train_cycle(
    model: nn.Module,
    train_one_epoch: OneEpochRunner,
    validate_one_epoch: OneEpochRunner,
    epochs: int,
    dump_model_filename: str,
    lr_scheduler,
    plot_fn: Optional[Callable[List[float], List[float], None]],
) -> Dict[str, List[float]]:
    best_val_loss = np.inf
    train_loss_hist, train_score_hist = [], []
    val_loss_hist, val_score_hist = [], []
    for epoch in range(epochs):
        train_loss, train_scores = train_one_epoch()
        val_loss, val_scores = validate_one_epoch()
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        if train_scores:
            train_score_hist.append(train_scores)
        if val_scores:
            val_score_hist.append(val_scores)
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

    result = dict(train_loss=train_loss_hist, val_loss=val_loss_hist)
    result.update(to_column_view(train_score_hist, prefix="train_"))
    result.update(to_column_view(val_score_hist, prefix="val_"))
    return result


@time_it("main_train_model, duration", logger.info)
def main_train_model(cfg: Config):
    ds = load_processed_dataset(to_absolute_path(cfg.train_dataset))
    vocab = torch.load(to_absolute_path(cfg.vocab_path))
    args = cfg.train
    train_dataset, val_dataset = ds.train_test_split(args.test_size)
    train_dataloader = make_text_dataloader(
        train_dataset, args.batch_size, vocab[PAD]
    )
    val_dataloader = make_text_dataloader(
        train_dataset, args.batch_size, vocab[PAD]
    )

    model = make_model(args.model, vocab)
    loss_fn = make_loss_fn(args.loss_fn)

    optimizer = make_optimizer(args.optimizer, model, args.learning_rate)
    device = get_device(args.gpu)

    score_functions = dict(
        accuracy=accuracy_from_predictions,
        recall=recall_from_predictions,
        precision=precision_from_predictions,
    )
    train_one_epoch = make_one_epoch_runner(
        args.model,
        model,
        loss_fn,
        train_dataloader,
        optimizer,
        device,
        is_train=True,
        score_functions=score_functions,
    )
    val_one_epoch = make_one_epoch_runner(
        args.model,
        model,
        loss_fn,
        val_dataloader,
        None,
        device,
        is_train=False,
        score_functions=score_functions,
    )

    lr_scheduler = make_lr_scheduler(optimizer, args.lr_scheduler)
    plot_fn = plot_train_val_loss if args.interactive else None
    results = train_cycle(
        model,
        train_one_epoch,
        val_one_epoch,
        args.epochs,
        ensure_path(args.dump_model),
        lr_scheduler,
        plot_fn,
    )
    save_train_report(results, ensure_path(args.report_path))
    run_visualization(
        to_absolute_path(cfg.prepare_data.train_file), args.report_path, args.dump_model
    )
