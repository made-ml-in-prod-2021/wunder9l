from torch.utils import data

from src.constants.consts import START_OF_SEQUENCE, END_OF_SEQUENCE
from src.utils.dataset_utils import VocabTransform


class MyTextDataset(data.Dataset):
    def __init__(self, texts, labels, transforms, tokenizer, vocab):
        self.texts = texts
        self.labels = labels
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.vocab = VocabTransform(vocab)

    def __getitem__(self, idx):
        sample = self.texts[idx]
        if self.transforms is not None:
            sample = self.transforms(sample)
        sample = [START_OF_SEQUENCE] + self.tokenizer(sample) + [END_OF_SEQUENCE]
        sample = self.vocab(sample)
        return {"text": sample, "label": self.labels[idx]}

    def __len__(self):
        return len(self.labels)
