from unittest.mock import patch, MagicMock

import numpy as np
from torch.utils.data import Dataset
from torchtext.vocab import Vocab

from src.data.make_dataset import clear_raw_dataset, load_raw_csv, make_datasets, read_datasets
from tests.utils.fixtures import raw_dataset_dataframe, raw_dataset_file, preprocessed_dataset_file
from tests.utils.helpers import generate_labels, generate_texts


def test_clear_raw_dataset(raw_dataset_dataframe):
    df = clear_raw_dataset(raw_dataset_dataframe)
    assert set(df.columns) == {"label", "text"}


@patch("src.data.make_dataset.clear_raw_dataset")
def test_load_raw_csv(mocked_func: MagicMock, raw_dataset_file):
    mocked_func.return_value = "expected"
    dataset = load_raw_csv(raw_dataset_file)
    mocked_func.assert_called_once()
    assert dataset == "expected"


def test_make_datasets():
    size = 1000
    labels_cnt = 2
    test_size = 0.3
    tokenizer_name = "basic_english"
    labels = generate_labels(labels_cnt, size)
    texts = generate_texts(size)

    train_dataset, test_dataset, vocab = make_datasets(
        labels, texts, test_size, pretrained_vectors=None, tokenizer_name=tokenizer_name
    )
    assert isinstance(train_dataset, Dataset)
    assert isinstance(test_dataset, Dataset)
    assert isinstance(vocab, Vocab)
    assert len(train_dataset) == (1 - test_size) * size
    assert len(test_dataset) == test_size * size
    sample = train_dataset[123]
    assert sample['label'] in (0, 1)
    assert all(isinstance(x, int) for x in sample['text'])
    assert all(x < len(vocab.itos) for x in sample['text'])


@patch("src.data.make_dataset.make_datasets")
def test_read_datasets(mocked_func: MagicMock, preprocessed_dataset_file):
    mocked_func.return_value = "expected"
    test_size = 0.45
    tokenizer_name = "basic_english"
    pretrained_vectors = 'glove.6B.300d'
    read_datasets(preprocessed_dataset_file, test_size, tokenizer_name, pretrained_vectors)
    mocked_func.assert_called_once()
    labels = mocked_func.call_args.args[0]
    texts = mocked_func.call_args.args[1]
    assert isinstance(labels, np.ndarray)
    assert all(x == 0 or x == 1 for x in labels)
    assert isinstance(texts, np.ndarray)
    assert test_size == mocked_func.call_args.args[2]
    assert pretrained_vectors == mocked_func.call_args.args[3]
    assert tokenizer_name == mocked_func.call_args.args[4]
