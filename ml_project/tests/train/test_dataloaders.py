from torch.utils.data import DataLoader

from src.constants.consts import PAD, DATA, TARGET
from src.data.make_dataset import read_datasets
from src.utils.dataset_utils import make_text_dataloader
from tests.utils.fixtures import preprocessed_dataset_file


def test_text_dataloader(preprocessed_dataset_file):
    batch_size = 2
    pretrained_vectors = None
    test_size = 0.4
    tokenizer_name = 'basic_english'
    train_dataset, val_dataset, vocab = read_datasets(
        preprocessed_dataset_file,
        test_size,
        tokenizer_name,
        pretrained_vectors,
    )
    train_dataloader = make_text_dataloader(
        train_dataset, batch_size, train_dataset.vocab[PAD]
    )
    assert isinstance(train_dataloader, DataLoader)
    batch_count = 0
    for batch in train_dataloader:
        assert DATA in batch
        assert TARGET in batch
        assert batch[DATA].shape[0] == 2
        assert batch[TARGET].shape == (2, )
        batch_count += 1
    assert batch_count == 3
