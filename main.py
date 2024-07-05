from pathlib import Path
import configargparse

from datasets import load_dataset

import torchtext
import torchmetrics

torchtext.disable_torchtext_deprecation_warning()

from dataset.dataset import LoaderConstructor

import wandb
import torch
import torch.nn as nn
import torch.optim as optim

from models.lstm import LSTM
from models.xlstm import xLSTM
from models.transformer import Transformer

from trainer import Trainer
from scheduler import ChainedScheduler


def get_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False

    config_path = Path(__file__).parent / "config.yaml"
    parser = configargparse.ArgumentParser(
        default_config_files=[config_path],
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )

    parser.add_argument("--model", type=str, default="xlstm")
    parser.add_argument(
        "--dataset", type=str, default="wikitext-2"
    )  # ptb_text_only, wikitext-2-v1, wikitext-103-v1
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=20)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min_words", type=int, default=200)
    parser.add_argument("--wandb", type=str2bool, default=False)
    parser.add_argument("--chained-scheduler", type=str2bool, default=False)
    parser.add_argument("--tokenizer", type=str, default="torchtext")
    return parser.parse_args()


def get_model(model, vocab_size, embed_dim, seq_len, output_dim, device):
    if model == "lstm":
        return LSTM(
            vocab_size=vocab_size,
            embedding_dim=embed_dim,
            hidden_dim=512,
            output_dim=output_dim,
            num_layers=2,
        )
    elif model == "transformer":
        return Transformer(
            vocab_size=vocab_size,
            seq_len=seq_len,
            embed_dim=embed_dim,
            output_dim=output_dim,
            num_layers=2,
            num_heads=8,
            dropout=0.1,
        )
    elif model == "xlstm":
        return xLSTM(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            seq_len=seq_len,
            out_features=output_dim,
            device=device,
        )
    else:
        raise ValueError(f"Model {model} not found")


if __name__ == "__main__":
    cfg = get_args()

    # Load the dataset
    dataset = load_dataset("wikitext", f"{cfg.dataset}-raw-v1")

    # Construct the dataloaders
    labels_sequence = True if cfg.model in ["transformer", "xlstm"] else False
    lc = LoaderConstructor(
        dataset=dataset,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
        min_words=cfg.min_words,
        tokenizer_type=cfg.tokenizer,
        labels_sequence=labels_sequence,
    )
    loaders = {}
    for loader in ["train", "validation", "test"]:
        loaders[loader] = lc.construct_loader(split=loader)

    input_size = loaders["train"].dataset.input_size
    vocab_size = lc.vocab_size
    output_size = lc.output_size
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the model
    model = get_model(
        model=cfg.model,
        vocab_size=vocab_size,
        embed_dim=cfg.embed_dim,
        seq_len=input_size,
        output_dim=output_size,
        device=device,
    )

    # Init wandb logger
    if cfg.wandb:
        wandb.init(project="text-generation", config=cfg)

    # Initialize the optimizer, loss function, and accuracy metric
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()
    accuracy = torchmetrics.Accuracy(
        task="multiclass", num_classes=output_size, top_k=5
    ).to(device)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        accuracy=accuracy,
        batch_size=cfg.batch_size,
        output_dim=output_size,
        wandb=cfg.wandb,
        device=device,
    )

    scheduler = None
    if cfg.chained_scheduler:
        warmup_steps = 4
        scheduler = ChainedScheduler(
            trainer.optimizer,
            T_0=(cfg.epochs - warmup_steps),
            T_mul=1,
            eta_min=cfg.lr / 10,
            gamma=0.5,
            max_lr=cfg.lr,
            warmup_steps=warmup_steps,
        )

    # Train the model
    best_valid_loss = float("inf")
    for epoch in range(cfg.epochs):
        if cfg.wandb:
            lr = trainer.optimizer.param_groups[0]["lr"]
            wandb.log({"Learning Rate": lr}, step=epoch)

        # Training
        model.train()
        trainer.train_validate_epoch(loaders["train"], epoch, "train")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = trainer.train_validate_epoch(
                loaders["validation"], epoch, "validation"
            )

        # Save the best model
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            torch.save(model.state_dict(), f"{cfg.model}_best.pt")
            print(f"Model improved, saving model")

        if scheduler:
            scheduler.step()

    # Load the best model
    model.load_state_dict(
        torch.load(f"{cfg.model}_best_{best_valid_loss}_{cfg.dataset}.pt")
    )

    # Test the model
    model.eval()
    with torch.no_grad():
        trainer.train_validate_epoch(loaders["test"], None, "test")

    exit()
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

        predictions = []
        for i in range(n_next_words):
            output = model(inputs)
            next_token = torch.argmax(output[:, -1, :])
            next_word_prediction = str(lc.tokenizer.decode(next_token))
            predictions.append(next_word_prediction)
            inputs = torch.cat([inputs[:, 1:], next_token.reshape(1, -1)], dim=1)

        print(f"Prediction: {text} {' '.join(predictions)}")
