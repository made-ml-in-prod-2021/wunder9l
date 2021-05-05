import random
import faker
import pandas as pd
import pytest


def make_dataset_sample(label_column, text_column) -> pd.DataFrame:
    fake = faker.Faker()
    columns = [
        {
            label_column: random.choice(["spam", "ham"]),
            text_column: fake.text(),
            "some": fake.name(),
            "other": fake.email(),
            "columns": fake.address(),
        }
        for _ in range(10)
    ]
    return pd.DataFrame(columns)


@pytest.fixture()
def raw_dataset_dataframe() -> pd.DataFrame:
    return make_dataset_sample("v1", "v2")


@pytest.fixture()
def raw_dataset_file(tmpdir, raw_dataset_dataframe) -> str:
    tmpfile = tmpdir.join("sample.csv")
    raw_dataset_dataframe.to_csv(tmpfile)
    return tmpfile


# @pytest.fixture()
def preprocessed_dataset_dataframe() -> pd.DataFrame:
    fake = faker.Faker()
    columns = [
        {
            "label": random.choice(["spam", "ham"]),
            "text": fake.text(),
        }
        for _ in range(10)
    ]
    return pd.DataFrame(columns)


@pytest.fixture()
def preprocessed_dataset_file(tmpdir) -> str:
    tmpfile = tmpdir.join("sample.csv")
    preprocessed_dataset_dataframe().to_csv(tmpfile)
    return tmpfile
