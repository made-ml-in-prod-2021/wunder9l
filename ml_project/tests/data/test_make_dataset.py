import io
import random

import pandas as pd
import pytest
import faker
from unittest.mock import patch, MagicMock

import src.data.make_dataset
from src.data.make_dataset import clear_raw_dataset, load_raw_csv


@pytest.fixture()
def raw_dataset_dataframe() -> pd.DataFrame:
    fake = faker.Faker()
    columns = [{
        "v1": random.choice(["spam", "ham"]),
        "v2": fake.text(),
        "some": fake.name(),
        "other": fake.email(),
        "columns": fake.address(),
    }]
    return pd.DataFrame(columns)


@pytest.fixture()
def raw_dataset_file(tmpdir, raw_dataset_dataframe) -> str:
    tmpfile = tmpdir.join("sample.csv")
    raw_dataset_dataframe.to_csv(tmpfile)
    return tmpfile


def test_clear_raw_dataset(raw_dataset_dataframe):
    dataset = clear_raw_dataset(raw_dataset_dataframe)
    assert set(dataset.columns) == {"label", "text"}


@patch("src.data.make_dataset.clear_raw_dataset")
def test_load_raw_csv(mocked_func: MagicMock, raw_dataset_file):
    mocked_func.return_value = "expected"
    dataset = load_raw_csv(raw_dataset_file)
    mocked_func.assert_called_once()
    assert dataset == "expected"
