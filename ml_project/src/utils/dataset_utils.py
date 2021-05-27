import collections
import os.path
from collections import Callable
from typing import List, Dict, Iterator
import numpy as np
import torch
import torchtext.vocab
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, Sampler

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
        texts = pad_sequence(texts, padding_value=padding_value, batch_first=False)
        target = [item[TARGET] for item in batch]
        return {DATA: texts, TARGET: torch.tensor(target)}

    return collate_fn


class BatchSampler(Sampler):
    def __init__(self, dataset, batch_size, pool_per_batch=100):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.pool_per_batch = pool_per_batch
        self.indices = [(i, len(item[DATA])) for i, item in enumerate(self.dataset)]

    def __iter__(self) -> Iterator:
        np.random.shuffle(self.indices)
        pooled_indices = []
        # create pool of indices with similar lengths
        for i in range(0, len(self.indices), self.batch_size * self.pool_per_batch):
            pooled_indices.extend(
                sorted(self.indices[i : i + self.batch_size * 100], key=lambda x: x[1])
            )

        self.pooled_indices = [x[0] for x in pooled_indices]
        self.index = 0
        return self

    def __next__(self):
        i = self.index
        if i >= len(self.pooled_indices):
            raise StopIteration
        self.index += self.batch_size
        return self.pooled_indices[i: i + self.batch_size]

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


def make_text_dataloader(dataset: Dataset, batch_size: int, padding_value: float):
    bucket_dataloader = DataLoader(
        dataset,
        batch_sampler=BatchSampler(dataset, batch_size),
        collate_fn=collate_batch(padding_value),
    )
    return bucket_dataloader


def ensure_path(path_to_file):
    dirname = os.path.dirname(path_to_file)
    os.makedirs(dirname, exist_ok=True)
    return path_to_file
