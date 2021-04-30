import pandas as pd
from sklearn.pipeline import Pipeline
from nltk.tokenize import wordpunct_tokenize

from src.constants.enums import TokenizerType
from src.utils.dataset_utils import FeaturesWithLabels
from src.utils.transform_adapter import TransformCallbackAdapter


def build_tokenizer(method: TokenizerType):
    if method == TokenizerType.WORD_PUNCTUATION:
        return TransformCallbackAdapter(wordpunct_tokenize)
    else:
        raise NotImplementedError(f"Tokenizer for {method} not implemented yet")


def build_transformers(
    need_lowercase: bool, tokenizer_method: TokenizerType
) -> Pipeline:
    steps = []
    if need_lowercase:
        steps.append(("lowercase", TransformCallbackAdapter(lambda x: x.lower())))
    steps.append(("tokenizer", build_tokenizer(tokenizer_method)))
    return Pipeline(steps)


def make_features(
    df: pd.DataFrame,
    need_lowercase: bool,
    tokenizer_method: TokenizerType = TokenizerType.WORD_PUNCTUATION,
) -> FeaturesWithLabels:
    labels = df.label.values.tolist()
    texts = df.text.values
    transformers = build_transformers(need_lowercase, tokenizer_method)
    texts = transformers.fit_transform(texts, labels)
    return FeaturesWithLabels(texts, labels)


def read_and_make_features(
    filename: str, need_lowercase: bool, tokenizer_method: TokenizerType
) -> FeaturesWithLabels:
    df = pd.read_csv(filename)
    return make_features(df, need_lowercase, tokenizer_method)
