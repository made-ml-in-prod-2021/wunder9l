# -*- coding: utf-8 -*-
import pickle
from collections import Counter

import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, List

import pandas as pd
import torch
from hydra.utils import to_absolute_path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchtext.data import get_tokenizer
from torchtext.vocab import Vocab

from src.config.config import PrepareDataArgs
from src.constants.consts import *
from src.data.text_dataset import MyTextDataset, TokenizedDataset
from src.utils.decorators import time_it

logger = logging.getLogger(APP_NAME)


def load_raw_csv(filename: str) -> pd.DataFrame:
    try:
        dataset = pd.read_csv(filename)
    except Exception as error:
        logger.error(
            f"Exception during loading raw dataset: {error.message if hasattr(error, 'message') else error}"
        )
        raise error
    logger.info("Raw dataset was successfully read")
    cleared_dataset = clear_raw_dataset(dataset)
    return cleared_dataset


def clear_raw_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    dataset.rename(columns={"v1": "label", "v2": "text"}, inplace=True)
    return dataset[["label", "text"]]


def save_dataset(ds: MyTextDataset, output_filepath: str) -> None:
    with open(output_filepath, "wb") as out:
        pickle.dump(ds.compacted_repr(), out)
    logger.info(f"Raw data processed to {output_filepath}")


def load_processed_dataset(filepath: str) -> TokenizedDataset:
    with open(filepath, "rb") as inp:
        ds = pickle.load(inp)
    logger.info(f"Tokenized dataset loaded from {filepath}")
    return ds


def normal_tensor_like(like: torch.Tensor) -> torch.Tensor:
    return torch.normal(mean=0.0, std=1.0, size=like.shape)


def build_vocab(
    train_texts,
    tokenizer,
    pretrained_vectors: Optional[str],
    vector_cache_dir: Optional[str],
):
    counter = Counter()
    for sample in train_texts:
        counter.update(tokenizer(sample))
    vocabulary = Vocab(
        counter,
        max_size=500000,
        vectors=pretrained_vectors,
        specials=[END_OF_LINE, PAD, UNKNOWN, START_OF_SEQUENCE, END_OF_SEQUENCE],
        vectors_cache=vector_cache_dir,
        unk_init=normal_tensor_like,
    )
    return vocabulary


def read_datasets(
    filename: str,
    test_size: float = 0.2,
    tokenizer_name: str = "basic_english",
    pretrained_vectors: str = "glove.6B.100d",
    vector_cache_dir: Optional[str] = ".vector_cache",
) -> (Dataset, Dataset, Vocab):
    df = pd.read_csv(filename)
    labels = df.label.values
    labels = LabelEncoder().fit_transform(labels)
    texts = df.text.values
    return make_datasets(
        labels, texts, test_size, pretrained_vectors, tokenizer_name, vector_cache_dir
    )


def make_datasets(
    labels,
    texts,
    test_size: float,
    pretrained_vectors: Optional[str],
    tokenizer_name: str,
    vector_cache_dir: str,
) -> Tuple[Dataset, Dataset, Vocab]:
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=RANDOM_SEED, shuffle=True
    )
    tokenizer = get_tokenizer(tokenizer_name)
    vocab = build_vocab(train_texts, tokenizer, pretrained_vectors, vector_cache_dir)
    transforms = None
    return (
        MyTextDataset(train_texts, train_labels, transforms, tokenizer, vocab),
        MyTextDataset(test_texts, test_labels, transforms, tokenizer, vocab),
        vocab,
    )


def make_dataset(
    df: pd.DataFrame, label_encoder: LabelEncoder, transforms, tokenizer, vocab
) -> MyTextDataset:
    labels = label_encoder.transform(df.label.values)
    texts = df.text.values
    ds = MyTextDataset(texts, labels, transforms, tokenizer, vocab)
    return ds


@time_it("prepare_data, duration", logger.info)
def prepare_data(args: PrepareDataArgs, train_output, test_output, vocab_output):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger.info("Processing raw data to final data set")
    train_df = load_raw_csv(to_absolute_path(args.train_file))
    test_df = load_raw_csv(to_absolute_path(args.test_file))

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df.label.values)

    tokenizer = get_tokenizer(args.tokenizer_name)
    vocab = build_vocab(
        train_df.text.values,
        tokenizer,
        args.pretrained_vectors,
        to_absolute_path(args.vectors_cache_directory),
    )
    logger.info(f"Save vocab into {vocab_output}")
    torch.save(vocab, vocab_output)

    train_ds = make_dataset(
        train_df, label_encoder, transforms=None, tokenizer=tokenizer, vocab=vocab
    )
    save_dataset(train_ds, train_output)

    test_ds = make_dataset(
        test_df, label_encoder, transforms=None, tokenizer=tokenizer, vocab=vocab
    )
    save_dataset(test_ds, test_output)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    prepare_data()
