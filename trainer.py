import wandb
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        accuracy,
        batch_size,
        output_dim,
        wandb,
        device,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.accuracy = accuracy
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.wandb = wandb
        self.device = device

        self.model.to(self.device)

    def metrics_to_wandb(self, split, loss, accuracy, loader_len, epoch=None):
        wandb.log(
            {
                f"{split}/loss": loss / loader_len,
                f"{split}/accuracy": accuracy,
            },
            step=epoch,
        )

    def train_validate_epoch(self, loader, epoch, split):
        # Reset metrics
        total_loss = 0
        self.accuracy.reset()
        for batch in tqdm(loader, total=len(loader)):
            # Forward pass
            inputs, labels = batch["input_ids"].to(self.device), batch["labels"].to(
                self.device
            )
            labels = labels.contiguous().view(-1)
            output = self.model(inputs).view(-1, self.output_dim)

            # Compute accuracy and loss
            self.accuracy.update(output, labels)
            loss = self.criterion(output, labels)

            # Backpropagation
            if split == "train":
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()

        if self.wandb:
            self.metrics_to_wandb(
                split, total_loss, self.accuracy.compute(), len(loader), epoch
            )
        print(
            f"[{split.upper()} {epoch}]: Loss: {total_loss / len(loader)}, Accuracy: {self.accuracy.compute()}"
        )
        return total_loss
