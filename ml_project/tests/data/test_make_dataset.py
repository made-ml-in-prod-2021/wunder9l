from unittest.mock import patch, MagicMock

from src.data.make_dataset import clear_raw_dataset, load_raw_csv
from tests.utils.fixtures import raw_dataset_dataframe, raw_dataset_file


def test_clear_raw_dataset(raw_dataset_dataframe):
    dataset = clear_raw_dataset(raw_dataset_dataframe)
    assert set(dataset.columns) == {"label", "text"}


@patch("src.data.make_dataset.clear_raw_dataset")
def test_load_raw_csv(mocked_func: MagicMock, raw_dataset_file):
    mocked_func.return_value = "expected"
    dataset = load_raw_csv(raw_dataset_file)
    mocked_func.assert_called_once()
    assert dataset == "expected"
