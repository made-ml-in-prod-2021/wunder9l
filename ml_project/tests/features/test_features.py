import pytest

from src.constants.enums import TokenizerType
from src.features.build_features import build_transformers, make_features
from src.utils.dataset_utils import FeaturesWithLabels
from src.utils.transform_adapter import TransformCallbackAdapter
from tests.utils.fixtures import preprocessed_dataset_dataframe


@pytest.mark.parametrize('need_lowercase', [False, True])
def test_build_transformers(need_lowercase):
    transformers = build_transformers(need_lowercase, TokenizerType.WORD_PUNCTUATION)
    assert len(transformers) == 1 + int(need_lowercase)
    assert isinstance(transformers[-1], TransformCallbackAdapter)


@pytest.mark.parametrize('need_lowercase', [False, True])
def test_build_features(need_lowercase, preprocessed_dataset_dataframe):
    features = make_features(preprocessed_dataset_dataframe, need_lowercase, TokenizerType.WORD_PUNCTUATION)
    assert isinstance(features, FeaturesWithLabels)
    assert len(features.labels) == len(features.features)
    assert all(isinstance(x, list) for x in features.features)
    for sample in features.features:
        assert all(isinstance(x, str) for x in sample)
        if need_lowercase:
            for word in sample:
                assert not any(c.isupper() for c in word)

# @patch("src.data.make_dataset.clear_raw_dataset")
# def test_load_raw_csv(mocked_func: MagicMock, raw_dataset_file):
#     mocked_func.return_value = "expected"
#     dataset = load_raw_csv(raw_dataset_file)
#     mocked_func.assert_called_once()
#     assert dataset == "expected"
