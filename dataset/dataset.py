from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

import re
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer


def create_rocstories_dataset(path):
    data_path = Path(path) / "data" / "rocstories.csv"
    df = pd.read_csv(data_path)
    df["text"] = (
        df["sentence1"]
        + " "
        + df["sentence2"]
        + " "
        + df["sentence3"]
        + " "
        + df["sentence4"]
        + " "
        + df["sentence5"]
    )
    train_data, temp_data = train_test_split(df["text"], test_size=0.3, random_state=42)
    test_data, val_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Convert the splits to the desired dictionary format
    train_list = [{"text": value} for value in train_data]
    val_list = [{"text": value} for value in val_data]
    test_list = [{"text": value} for value in test_data]

    # Combine into a single dictionary
    data_splits = {"train": train_list, "validation": val_list, "test": test_list}
    return data_splits


def create_alicewonderland_dataset(path):
    data_path = Path(path) / "data" / "alice_in_wonderland.txt"
    with open(data_path, "r") as file:
        text = file.read()
    text = text.split("\n")

    # Remove lines with less than 10 words (arbitrary threshold) because they are not useful
    text = [line.strip() for line in text if len(line.strip().split()) > 10]

    # Split the text into training, validation and test sets
    train_data, temp_data = train_test_split(text, test_size=0.3, random_state=42)
    test_data, val_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Convert the splits to the desired dictionary format
    train_list = [{"text": value} for value in train_data]
    val_list = [{"text": value} for value in val_data]
    test_list = [{"text": value} for value in test_data]

    # Combine into a single dictionary
    data_splits = {"train": train_list, "validation": val_list, "test": test_list}
    return data_splits


class TextDataset(Dataset):
    def __init__(self, encodings, labels_sequence):
        self.encodings = encodings
        self.input_size = len(encodings["input_ids"][0])
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
    def __init__(self, max_length, special_tokens_in_target):
        self.tokenizer = get_tokenizer("basic_english")
        self.vocab_size = None
        self.output_size = None
        self.max_length = max_length
        self.special_tokens_in_target = special_tokens_in_target

    def create_tokens(self, dataset):
        def clean_text(text):
            cleaned_text = "".join([c if c.isalpha() else " " for c in text])
            cleaned_text = re.sub(r"\b[b-hj-z]\b", "", cleaned_text)
            return cleaned_text.strip()

        tokenised_samples = []
        for sample in dataset:
            clean_sample = clean_text(sample["text"])
            tokenised_samples.append(self.tokenizer(clean_sample))

        return tokenised_samples

    def create_vocab(self, tokenised_samples, min_freq):
        self.train_vocab = build_vocab_from_iterator(
            tokenised_samples,
            min_freq=min_freq,
            specials=["<pad>", "<oov>"],
            special_first=True,
        )
        self.vocab_size = len(self.train_vocab)
        if self.special_tokens_in_target:
            self.target_vocab = self.train_vocab
            self.output_size = self.vocab_size
        else:
            self.target_vocab = build_vocab_from_iterator(
                tokenised_samples, min_freq=min_freq
            )
            self.output_size = len(self.target_vocab)

        return self.vocab_size, self.output_size

    def pad_sequences(self, tokenised_samples):
        # Pad the sequences to the max_length
        for i, sample in enumerate(tokenised_samples):
            if len(sample) < self.max_length:
                tokenised_samples[i] = ["<pad>"] * (
                    self.max_length - len(sample)
                ) + sample
        return tokenised_samples

    def create_subsequences(self, tokenised_samples, stride=20):
        # Create subsequences with a fixed length and sliding window
        sequences = []
        for sample in tokenised_samples:
            current = 0
            while current + self.max_length <= len(sample):
                sequences.append(sample[current : current + self.max_length])
                current += stride
        return sequences

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

    def encode(self, samples):
        stoi = self.train_vocab.get_stoi()
        samples = [
            [stoi[token] if token in stoi else stoi["<oov>"] for token in sample]
            for sample in samples
        ]
        return torch.tensor(samples)

    def decode(self, tokens, target=True):
        if target:
            token_dict = self.target_vocab.get_itos()
        else:
            token_dict = self.train_vocab.get_itos()
        return [token_dict[token] for token in tokens]


class LoaderConstructor:
    def __init__(
        self,
        dataset,
        batch_size,
        max_length,
        min_freq=3,
        labels_sequence=False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length + 1  # Add 1 for the labels
        self.min_freq = min_freq
        self.labels_sequence = labels_sequence

        self.tokenizer = TorchtextTokenizer(
            max_length=self.max_length,
            special_tokens_in_target=self.labels_sequence,
        )

    def construct_loader(self, split):
        encodings = self.torchtext_tokenize(split)

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
                tokenised_samples, self.min_freq
            )

        # Pad the sequences to the max_length
        tokenised_samples = self.tokenizer.pad_sequences(tokenised_samples)

        # Create subsequences with a fixed length and sliding window
        sequences = self.tokenizer.create_subsequences(tokenised_samples)

        # Tokenize the dataset
        encodings = self.tokenizer.tokenize(sequences)
        return encodings
