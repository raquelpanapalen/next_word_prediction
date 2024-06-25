from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, encodings, vocab_size, labels_sequence):
        self.encodings = encodings
        self.vocab_size = vocab_size
        self.input_size = len(encodings["input_ids"][0]) - 1
        self.labels_sequence = labels_sequence

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        sample = self.encodings["input_ids"][idx]
        item = {
            "input_ids": sample[:-1],
            "labels": sample[1:] if self.labels_sequence else sample[-1],
        }
        return item


class LoaderConstructor:
    def __init__(
        self, dataset, batch_size, max_length, min_words, labels_sequence=False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length
        self.min_words = min_words
        self.labels_sequence = labels_sequence

        # Load the tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "google-bert/bert-large-uncased", padding_side="left"
        )
        self.vocab_size = self.tokenizer.vocab_size

    def construct_loader(self, split):

        def remove_non_alpha(text):
            return "".join([c if c.isalpha() else " " for c in text])

        # Tokenize the dataset
        dataset = [
            remove_non_alpha(sample["text"])
            for sample in self.dataset[split]
            if len(sample["text"]) > self.min_words
        ]
        encodings = self.tokenizer(
            dataset,
            truncation=True,
            padding="max_length",
            max_length=self.max_length + 1,
            add_special_tokens=False,
            return_tensors="pt",
        )
        dataset = TextDataset(
            encodings=encodings,
            vocab_size=self.vocab_size,
            labels_sequence=self.labels_sequence,
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        return loader
