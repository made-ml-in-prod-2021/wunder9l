# -*- coding: utf-8 -*-
from collections import Counter

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchtext.data import get_tokenizer
from torchtext.vocab import Vocab
from src.constants.consts import *
from src.data.text_dataset import MyTextDataset
from src.utils.decorators import time_it

logger = logging.getLogger(__name__)


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


def save_to_output(data: pd.DataFrame, output_filepath: str) -> None:
    data.to_csv(output_filepath, index=False)


def build_vocab(train_texts, tokenizer, pretrained_vectors, vector_cache_dir: str):
    counter = Counter()
    for sample in train_texts:
        counter.update(tokenizer(sample))
    vocabulary = Vocab(
        counter,
        max_size=500000,
        vectors=pretrained_vectors,
        specials=[END_OF_LINE, PAD, UNKNOWN, START_OF_SEQUENCE, END_OF_SEQUENCE],
        vectors_cache=vector_cache_dir,
    )
    return vocabulary


def read_datasets(
    filename: str,
    test_size: float = 0.2,
    tokenizer_name: str = "basic_english",
    pretrained_vectors: str = "glove.6B.100d",
    vector_cache_dir: str = ".vector_cache",
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
    pretrained_vectors: str,
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


@time_it("prepare_data, duration", logger.info)
def prepare_data(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger.info("Processing raw data to final data set")
    loaded = load_raw_csv(input_filepath)
    save_to_output(loaded, output_filepath)
    logger.info(f"Raw data processed from {input_filepath} to {output_filepath}")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    prepare_data()
