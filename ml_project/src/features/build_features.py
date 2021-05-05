import pandas as pd
from sklearn.pipeline import Pipeline
from nltk.tokenize import wordpunct_tokenize

from src.constants.consts import END_OF_LINE
from src.constants.enums import TokenizerType
from src.utils.dataset_utils import FeaturesWithLabels
from src.utils.transform_adapter import TransformCallbackAdapter

from torchtext.vocab import Vocab, Vectors
import nltk
from collections import Counter
from sklearn.model_selection import train_test_split



def build_tokenizer(method: TokenizerType):
    if method == TokenizerType.WORD_PUNCTUATION:
        return TransformCallbackAdapter(wordpunct_tokenize)
    else:
        raise NotImplementedError(f"Tokenizer for {method} not implemented yet")



def build_vocabulary(train_tokens: [[str]], end_with_eol=False):
    counter = Counter()
    for sample in train_tokens:
        counter.update(sample)
    if end_with_eol:
        counter.update([END_OF_LINE] * len(train_tokens))



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


from torchtext.datasets import IMDB
train_iter, test_iter = IMDB(split=('train', 'test'))

from torchtext.data.utils import get_tokenizer
tokenizer = get_tokenizer('basic_english')
from collections import Counter
from torchtext.vocab import Vocab

train_iter = IMDB(split='train')
counter = Counter()
for (label, line) in train_iter:
    counter.update(tokenizer(line))
vocab = Vocab(counter, min_freq=10, specials=('<unk>', '<BOS>', '<EOS>', '<PAD>'))

text_transform = lambda x: [vocab['<BOS>']] + [vocab[token] for token in tokenizer(x)] + [vocab['<EOS>']]
label_transform = lambda x: 1 if x == 'pos' else 0

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def collate_batch(batch):
   label_list, text_list = [], []
   for (_label, _text) in batch:
        label_list.append(label_transform(_label))
        processed_text = torch.tensor(text_transform(_text))
        text_list.append(processed_text)
   return torch.tensor(label_list), pad_sequence(text_list, padding_value=3.0)

train_iter = IMDB(split='train')
train_dataloader = DataLoader(list(train_iter), batch_size=8, shuffle=True,
                              collate_fn=collate_batch)
import random

train_iter = IMDB(split='train')
train_list = list(train_iter)
batch_size = 8  # A batch size of 8

def batch_sampler():
    indices = [(i, len(tokenizer(s[1]))) for i, s in enumerate(train_list)]
    random.shuffle(indices)
    pooled_indices = []
    # create pool of indices with similar lengths
    for i in range(0, len(indices), batch_size * 100):
        pooled_indices.extend(sorted(indices[i:i + batch_size * 100], key=lambda x: x[1]))

    pooled_indices = [x[0] for x in pooled_indices]

    # yield indices for current batch
    for i in range(0, len(pooled_indices), batch_size):
        yield pooled_indices[i:i + batch_size]

bucket_dataloader = DataLoader(train_list, batch_sampler=batch_sampler(),
                               collate_fn=collate_batch)