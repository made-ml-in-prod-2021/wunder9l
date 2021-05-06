import collections
from collections import Callable
from typing import List, Dict
import numpy as np
import torch
import torchtext.vocab
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from src.constants.consts import DATA, TARGET


class FeaturesWithLabels(object):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    # def split(self, train_size):


class VocabTransform(object):
    def __init__(self, vocab: torchtext.vocab.Vocab):
        self.inner_vocab = vocab

    def __call__(self, list_of_tokens: List[str]):
        return [self.inner_vocab[token] for token in list_of_tokens]


def collate_batch(padding_value: float) -> Callable:
    def collate_fn(batch: List[Dict]):
        texts = [torch.tensor(item[DATA]) for item in batch]
        texts = pad_sequence(texts, padding_value=padding_value, batch_first=True)
        target = [item[TARGET] for item in batch]
        return {DATA: texts, TARGET: torch.tensor(target)}

    return collate_fn


def batch_sampler(dataset, batch_size, pool_per_batch=100):
    indices = []
    for i in range(len(dataset)):
        indices.append((i, len(dataset[i])))
    np.random.shuffle(indices)
    pooled_indices = []
    # create pool of indices with similar lengths
    for i in range(0, len(indices), batch_size * pool_per_batch):
        pooled_indices.extend(
            sorted(indices[i : i + batch_size * 100], key=lambda x: x[1])
        )

    pooled_indices = [x[0] for x in pooled_indices]

    # yield indices for current batch
    for i in range(0, len(pooled_indices), batch_size):
        yield pooled_indices[i: i + batch_size]


def make_text_dataloader(dataset: Dataset, batch_size: int, padding_value: float):
    bucket_dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler(dataset, batch_size),
        collate_fn=collate_batch(padding_value),
    )
    return bucket_dataloader
