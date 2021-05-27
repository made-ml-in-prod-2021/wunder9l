from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils import data
from torchtext.vocab import Vocab

from src.constants.consts import START_OF_SEQUENCE, END_OF_SEQUENCE, DATA, TARGET
from src.utils.dataset_utils import VocabTransform


class MyTextDataset(data.Dataset):
    def __init__(self, texts, labels, transforms, tokenizer, vocab: Vocab):
        self.texts = texts
        self.labels = labels
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.vocab_transform = VocabTransform(vocab)

    def __getitem__(self, idx):
        sample = self.texts[idx]
        if self.transforms is not None:
            sample = self.transforms(sample)
        sample = [START_OF_SEQUENCE] + self.tokenizer(sample) + [END_OF_SEQUENCE]
        sample = self.vocab_transform(sample)
        return {DATA: sample, TARGET: self.labels[idx]}

    def __len__(self):
        return len(self.labels)

    def compacted_repr(self):
        tokens = [self[idx][DATA] for idx in range(len(self))]
        return TokenizedDataset(tokens, self.labels)


class TokenizedDataset(data.Dataset):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __getitem__(self, idx):
        return {DATA: self.tokens[idx], TARGET: self.labels[idx]}

    def __len__(self):
        return len(self.labels)

    def train_test_split(self, test_size: float, seed: int = 42) -> Tuple:
        """Splits dataset to train/test"""
        train_tokens, test_tokens, train_labels, test_labels = train_test_split(
            self.tokens, self.labels, test_size=test_size, random_state=seed
        )
        return (
            TokenizedDataset(train_tokens, train_labels),
            TokenizedDataset(test_tokens, test_labels),
        )
