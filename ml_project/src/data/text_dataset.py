from torch.utils import data

from src.constants.consts import START_OF_SEQUENCE, END_OF_SEQUENCE, DATA, TARGET
from src.utils.dataset_utils import VocabTransform


class MyTextDataset(data.Dataset):
    def __init__(self, texts, labels, transforms, tokenizer, vocab):
        self.texts = texts
        self.labels = labels
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.vocab_transform = VocabTransform(vocab)

    def __getitem__(self, idx):
        sample = self.texts[idx]
        if self.transforms is not None:
            sample = self.transforms(sample)
        sample = [START_OF_SEQUENCE] + self.tokenizer(sample) + [END_OF_SEQUENCE]
        sample = self.vocab_transform(sample)
        return {DATA: sample, TARGET: self.labels[idx]}

    def __len__(self):
        return len(self.labels)
