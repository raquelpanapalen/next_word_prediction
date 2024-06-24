from pathlib import Path
import configargparse

from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    BertGenerationTokenizer,
    BertTokenizer,
)
from datasets import load_dataset

import torchtext
import torchmetrics

torchtext.disable_torchtext_deprecation_warning()

from torch.utils.data import DataLoader
from dataset.dataset import TextDataset

from tqdm import tqdm

import wandb

import math
import torch
import torch.nn as nn
import torch.optim as optim
from models.lstm import LSTM


def get_args():
    config_path = Path(__file__).parent / "config.yaml"
    parser = configargparse.ArgumentParser(
        default_config_files=[config_path],
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )

    parser.add_argument("--model", type=str, default="xlstm")
    parser.add_argument(
        "--dataset", type=str, default="wikitext"
    )  # ptb_text_only, wikitext-2-v1, wikitext-103-v1
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min_words", type=int, default=200)
    return parser.parse_args()


class LoaderConstructor:
    def __init__(self, dataset, batch_size, max_length, min_words):
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_length = max_length
        self.min_words = min_words

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
        dataset = TextDataset(encodings=encodings, vocab_size=self.vocab_size)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        return loader


if __name__ == "__main__":
    cfg = get_args()

    # Load the dataset
    dataset = load_dataset(cfg.dataset, "wikitext-103-raw-v1")

    # Construct the dataloaders
    lc = LoaderConstructor(dataset, cfg.batch_size, cfg.max_length, cfg.min_words)
    loaders = {}
    for loader in ["train", "validation", "test"]:
        loaders[loader] = lc.construct_loader(split=loader)

    vocab_size = lc.vocab_size
    input_size = loaders["train"].dataset.input_size  # or max_length
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the model
    model = LSTM(
        vocab_size=vocab_size,
        embedding_dim=input_size,
        hidden_dim=512,
        num_layers=1,
    )
    model.to(device)

    # Init wandb logger
    wandb.init(project="text-generation", config=cfg)

    def metrics_to_wandb(split, loss, accuracy, loader_len, epoch=None):
        wandb.log(
            {
                f"{split}/loss": loss / loader_len,
                f"{split}/accuracy": accuracy,
            },
            step=epoch,
        )

    # Initialize the optimizer, loss function, and accuracy metric
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()
    accuracy = torchmetrics.Accuracy(
        task="multiclass", num_classes=vocab_size, top_k=5
    ).to(device)

    verbose = False

    # Train the model
    best_valid_loss = float("inf")
    for epoch in range(cfg.epochs):
        train_loss = 0
        accuracy.reset()
        model.train()
        hidden = model.init_hidden(cfg.batch_size, device)
        for batch in tqdm(loaders["train"], total=len(loaders["train"])):
            # Forward pass
            optimizer.zero_grad()
            inputs, labels = batch["input_ids"].to(device), batch["labels"].to(device)
            hidden = model.detach_hidden(hidden)
            output, hidden = model(inputs, hidden)

            # Compute accuracy and loss
            accuracy.update(output[:, -1, :], labels)
            loss = criterion(output[:, -1, :], labels)

            # Backpropagation
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        metrics_to_wandb(
            "train", train_loss, accuracy.compute(), len(loaders["train"]), epoch
        )
        print(
            f"[TRAIN {epoch}/{cfg.epochs}]: Loss: {train_loss / len(loaders['train'])}, Accuracy: {accuracy.compute()}"
        )

        # Validation
        val_loss = 0
        accuracy.reset()
        model.eval()
        hidden = model.init_hidden(cfg.batch_size, device)
        with torch.no_grad():
            for batch in tqdm(loaders["validation"], total=len(loaders["validation"])):
                inputs, labels = batch["input_ids"].to(device), batch["labels"].to(
                    device
                )
                hidden = model.detach_hidden(hidden)
                output, hidden = model(inputs, hidden)
                accuracy.update(output[:, -1, :], labels)
                loss = criterion(output[:, -1, :], labels)
                val_loss += loss.item()

        metrics_to_wandb(
            "validation",
            val_loss,
            accuracy.compute(),
            len(loaders["validation"]),
            epoch,
        )
        print(
            f"[VALID {epoch}/{cfg.epochs}]: Loss: {val_loss / len(loaders['validation'])}, Accuracy: {accuracy.compute()}"
        )

        # Save the best model
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")
            print(f"Model improved, saving model")

    # Load the best model
    model.load_state_dict(torch.load("best_model.pt"))

    # Test the model
    test_loss = 0
    accuracy.reset()
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loaders["test"], total=len(loaders["test"])):
            inputs, labels = batch["input_ids"].to(device), batch["labels"].to(device)
            hidden = model.detach_hidden(hidden)
            output, hidden = model(inputs, hidden)
            loss = criterion(output[:, -1, :], labels)
            accuracy.update(output[:, -1, :], labels)
            test_loss += loss.item()
    metrics_to_wandb("test", test_loss, accuracy.compute(), len(loaders["test"]))
    print(
        f"[TEST]: Loss: {test_loss / len(loaders['test'])}, Accuracy: {accuracy.compute()}"
    )

    # Predict next word
    texts = [
        "The quick brown fox has",
        "I can't wait to go to",
        "The capital of France is",
        "The best way to learn is to",
    ]
    n_next_words = 5

    # Tokenize the texts
    for text in texts:
        tokens = lc.tokenizer(
            text, padding="max_length", max_length=cfg.max_length, return_tensors="pt"
        )
        inputs = tokens["input_ids"].to(device)

        hidden = model.init_hidden(batch_size=1, device=device)
        predictions = []
        for i in range(n_next_words):
            output, hidden = model(inputs, hidden)
            next_token = torch.argmax(output[:, -1, :])
            next_word_prediction = lc.tokenizer.decode(next_token)
            predictions.append(next_word_prediction)
            inputs = torch.cat([inputs[:, 1:], next_token.reshape(1, -1)], dim=1)

        print(f"Prediction: {text} {' '.join(predictions)}")
