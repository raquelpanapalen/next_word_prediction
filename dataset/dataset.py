import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

import re
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer


class TextDataset(Dataset):
    def __init__(self, encodings, labels_sequence):
        self.encodings = encodings
        self.input_size = len(encodings["input_ids"][0]) - 1
        self.labels_sequence = labels_sequence

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        input_ids = self.encodings["input_ids"][idx]
        labels = self.encodings["labels"][idx]
        item = {
            "input_ids": input_ids,
            "labels": labels if self.labels_sequence else labels[-1],
        }
        return item


class TorchtextTokenizer:
    def __init__(self):
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab_size = None
        self.output_size = None

    def create_tokens(self, dataset):
        def clean_text(text):
            return re.sub(r"[^a-zA-Z,. ]", "", text).strip()

        tokenised_samples = []
        for sample in dataset:
            clean_sample = clean_text(sample["text"])
            tokenised_samples.append(self.tokenizer(clean_sample))

        # Tokenize the dataset
        # tokenised_samples = list(map(self.tokenizer, dataset))

        return tokenised_samples

    def create_vocab(self, tokenised_samples):
        self.train_vocab = build_vocab_from_iterator(
            tokenised_samples,
            min_freq=3,
            specials=["<pad>", "<oov>"],
            special_first=True,
        )
        self.vocab_size = len(self.train_vocab)
        self.target_vocab = build_vocab_from_iterator(tokenised_samples, min_freq=3)
        self.output_size = len(self.target_vocab)

        return self.vocab_size, self.output_size

    def tokenize(self, tokenised_samples):
        # Convert the sequences to index tensors
        train_stoi = self.train_vocab.get_stoi()
        target_stoi = self.target_vocab.get_stoi()
        encodings = {"input_ids": [], "labels": []}
        for sequence in tokenised_samples:
            # Only add sequences that have a target token (no <pad> or <oov> tokens)
            if sequence[-1] in target_stoi:
                input_ids = torch.tensor(
                    [
                        (
                            self.train_vocab[token]
                            if token in train_stoi
                            else self.train_vocab["<oov>"]
                        )
                        for token in sequence[:-1]
                    ]
                )
                labels = torch.cat(
                    (
                        input_ids[1:],
                        torch.tensor(self.target_vocab[sequence[-1]]).unsqueeze(0),
                    )
                )
                encodings["input_ids"].append(input_ids)
                encodings["labels"].append(labels)

        return encodings

    def decode(self, tokens):
        return " ".join([self.target_vocab.itos[token] for token in tokens])


class LoaderConstructor:
    def __init__(
        self,
        dataset,
        batch_size,
        max_length,
        min_words,
        tokenizer_type,
        labels_sequence=False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length + 1  # Add 1 for the labels
        self.min_words = min_words
        self.tokenizer_type = tokenizer_type
        self.labels_sequence = labels_sequence

        # Load the tokenizer
        if tokenizer_type == "torchtext":
            self.tokenizer = TorchtextTokenizer()
        elif tokenizer_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(
                "google-bert/bert-large-uncased", padding_side="left"
            )
            self.vocab_size = self.tokenizer.vocab_size
            self.output_size = self.vocab_size
        else:
            raise ValueError(f"Tokenizer {tokenizer_type} not found")

    def construct_loader(self, split):
        if self.tokenizer_type == "torchtext":
            encodings = self.torchtext_tokenize(split)
        elif self.tokenizer_type == "bert":
            encodings = self.bert_tokenize(split)

        dataset = TextDataset(
            encodings=encodings,
            labels_sequence=self.labels_sequence,
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        return loader

    def torchtext_tokenize(self, split):
        tokenised_samples = self.tokenizer.create_tokens(self.dataset[split])

        # Build the vocabulary
        if split == "train":
            self.vocab_size, self.output_size = self.tokenizer.create_vocab(
                tokenised_samples
            )

        # Pad the sequences to the max_length
        for i, sample in enumerate(tokenised_samples):
            if len(sample) < self.max_length:
                tokenised_samples[i] = ["<pad>"] * (
                    self.max_length - len(sample)
                ) + sample

        # Create subsequences with a fixed length and sliding window
        stride = 30
        sequences = []
        for sample in tokenised_samples:
            current = 0
            while current + self.max_length < len(sample):
                sequences.append(sample[current : current + self.max_length])
                current += stride

        # Tokenize the dataset
        encodings = self.tokenizer.tokenize(sequences)
        return encodings

    # We won't be using this function
    def bert_tokenize(self, split):

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
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors="pt",
        )
        output = {
            "input_ids": encodings["input_ids"][:, :-1],
            "labels": encodings["input_ids"][:, 1:],  # Shift the labels
        }

        return output
