import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

from src.data.dataset import MultiviewActionDataset
from src.models.multiview_action_recognition_model import (
    MultiviewActionRecognitionModel,
)
from utils.logger import setup_logger
from utils.helper import (
    create_run_directory, save_checkpoint, load_config, save_config
)


class Trainer:
    """
    Trainer class for the multiview action recognition model.

    Attributes
    ----------
    train_config : dict
        Configuration for the training process (optimizer, scheduler,
        early stopping, etc.).
    model_config : dict
        Configuration for the model (architecture hyperparameters, etc.).
    device : torch.device
        Device to run the training on (CPU or CUDA).
    logger : logging.Logger
        Logger instance for logging information.
    model : nn.Module
        MultiviewActionRecognitionModel instance.
    optimizer : torch.optim.Optimizer
        Optimizer for training the model.
    criterion : nn.Module
        Loss function.
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    val_loader : torch.utils.data.DataLoader
        DataLoader for the validation dataset.
    scheduler : torch.optim.lr_scheduler.ReduceLROnPlateau
        Learning rate scheduler for adaptive LR changes.
    early_stopping_enabled : bool
        Flag indicating whether early stopping is enabled.
    early_stopping_patience : int
        Number of epochs without improvement before stopping.
    """

    def __init__(
            self, train_config: dict, model_config: dict, run_dir: str
    ) -> None:
        """
        Initialize the Trainer class with the provided configuration.

        Parameters
        ----------
        train_config : dict
            Training-related configuration dictionary.
        model_config : dict
            Model-related configuration dictionary.
        run_dir : str
            Directory for the current training run (for logs/checkpoints).
        """
        self.train_config = train_config
        self.model_config = model_config
        self.run_dir = run_dir

        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")

        # Setup logger to log into run_dir/training.log
        self.logger = setup_logger("training_log", os.path.join(
            run_dir,
            "training.log"
        ))

        # Check if wandb is enabled
        self.use_wandb = self.train_config.get("wandb", "no").lower() == "yes"
        if self.use_wandb:
            wandb.init(project="multiview-action-recognition", config={
                "train": self.train_config,
                "model": self.model_config
            })
            self.logger.info("WandB logging is enabled.")
        else:
            self.logger.info("WandB logging is disabled.")

        # Early Stopping
        self.early_stopping_enabled = (
            str(self.train_config.get("early_stopping", "no")).lower() == "yes"
        )
        self.early_stopping_patience = int(
            self.train_config.get("early_stopping_patience", 10)
        )
        self.no_improvement_count = 0
        self.best_val_loss = float("inf")

        # Initialize the model
        self.model = MultiviewActionRecognitionModel(
            num_heads=self.model_config["num_heads"],
            num_transformer_layers=self.model_config["num_transformer_layers"],
            num_classes=self.model_config["num_classes"],
        ).to(self.device)

        # Initialize the optimizer (AdamW) with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.train_config["learning_rate"],
            weight_decay=self.train_config.get("weight_decay", 0.0),
        )

        self.criterion = nn.CrossEntropyLoss()

        # Prepare data loaders
        self.train_loader, self.val_loader = self.get_dataloaders()

        # LR Scheduler (ReduceLROnPlateau) with patience
        lr_scheduler_patience = self.train_config.get(
            "lr_scheduler_patience",
            5
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=lr_scheduler_patience,
            factor=0.1,
            verbose=True
        )

    def get_dataloaders(self) -> tuple:
        """
        Initialize the training and validation data loaders.

        Returns
        -------
        tuple
            A tuple containing the training DataLoader and validation
            DataLoader.
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        seq_len = self.train_config.get("seq_len", 50)

        train_dataset = MultiviewActionDataset(
            data_dir="data/processed/train",
            transform=transform,
            seq_len=seq_len
        )
        val_dataset = MultiviewActionDataset(
            data_dir="data/processed/val",
            transform=transform,
            seq_len=seq_len
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_config["batch_size"],
            shuffle=True,
            num_workers=4
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.train_config["batch_size"],
            shuffle=False,
            num_workers=4
        )

        return train_loader, val_loader

    def train_one_epoch(self, epoch: int) -> float:
        """
        Train the model for one epoch.

        Parameters
        ----------
        epoch : int
            The current epoch number.

        Returns
        -------
        float
            The average training loss for this epoch.
        """
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch_idx, (views, actions) in enumerate(self.train_loader):
            views, actions = views.to(self.device), actions.to(self.device)

            # Forward pass
            outputs = self.model(views)
            loss = self.criterion(outputs, actions)

            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            total_predictions += actions.size(0)
            correct_predictions += (predicted == actions).sum().item()
            running_loss += loss.item()

            # Current LR from the scheduler
            current_lr = self.scheduler.get_last_lr()[0]

            if batch_idx % self.train_config["log_interval"] == 0:
                accuracy = 100.0 * correct_predictions / total_predictions
                self.logger.info(
                    f"Epoch [{epoch+1}/{self.train_config['num_epochs']}], "
                    f"Batch [{batch_idx}/{len(self.train_loader)}], "
                    f"Loss: {loss.item():.4f}, "
                    f"Accuracy: {accuracy:.2f}%, LR: {current_lr:.6f}"
                )

                # Log to wandb if enabled
                if self.use_wandb:
                    wandb.log({
                        "train_loss": loss.item(),
                        "train_accuracy": accuracy,
                        "learning_rate": current_lr
                    })

        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = 100.0 * correct_predictions / total_predictions

        self.logger.info(
            f"Epoch [{epoch+1}/{self.train_config['num_epochs']}], "
            f"Train Loss: {epoch_loss:.4f}, "
            f"Train Accuracy: {epoch_accuracy:.2f}%, "
            f"LR: {current_lr:.6f}"
        )

        return epoch_loss

    def validate(self) -> tuple:
        """
        Validate the model on the validation set.

        Returns
        -------
        tuple
            A tuple containing the validation loss and validation accuracy.
        """
        self.model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for views, actions in self.val_loader:
                views, actions = views.to(self.device), actions.to(self.device)

                outputs = self.model(views)
                loss = self.criterion(outputs, actions)

                _, predicted = torch.max(outputs, 1)
                total_predictions += actions.size(0)
                correct_predictions += (predicted == actions).sum().item()
                val_loss += loss.item()

        val_loss /= len(self.val_loader)
        val_accuracy = 100.0 * correct_predictions / total_predictions

        self.logger.info(
            f"Validation Loss: {val_loss:.4f}, "
            f"Validation Accuracy: {val_accuracy:.2f}%"
        )
        return val_loss, val_accuracy

    def train(self) -> None:
        """
        Full training loop with validation, checkpointing, LR scheduling,
        and optional early stopping.
        """
        for epoch in range(self.train_config["num_epochs"]):
            # Train for one epoch
            _ = self.train_one_epoch(epoch)

            # Validate the model
            val_loss, val_accuracy = self.validate()

            # Step the scheduler
            self.scheduler.step(val_loss)

            # Check if validation loss improved
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.no_improvement_count = 0

                # Save best checkpoint
                checkpoint_path = os.path.join(
                    self.run_dir,
                    f"best_model_epoch_{epoch+1}.pth"
                )
                save_checkpoint(
                    checkpoint_path, self.model,
                    self.optimizer, val_loss
                )
                self.logger.info(f"Best checkpoint saved at {checkpoint_path}")
            else:
                self.no_improvement_count += 1

            # Log validation metrics to wandb if enabled
            if self.use_wandb:
                current_lr = self.scheduler.get_last_lr()[0]
                wandb.log({
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "learning_rate": current_lr
                })

            # Early stopping check
            if self.early_stopping_enabled:
                if self.no_improvement_count >= self.early_stopping_patience:
                    self.logger.info(
                        "Early stopping triggered due to no improvement in "
                        f"validation loss for {self.no_improvement_count} "
                        "consecutive epochs."
                    )
                    break


def main() -> None:
    """
    Main entry point for training the multiview action recognition model.
    """
    # Default configuration with separate 'model' and 'train' sections
    default_config = {
        "model": {
            "num_heads": 4,
            "num_transformer_layers": 2,
            "num_classes": 10,
        },
        "train": {
            "batch_size": 8,
            "num_epochs": 50,
            "learning_rate": 1e-4,
            "seq_len": 50,
            "log_interval": 10,
            "wandb": "no",
            "early_stopping": "no",
            "early_stopping_patience": 10,
            "lr_scheduler_patience": 5,
            "weight_decay": 0.0,
            "output_dir": "output/models",
        }
    }

    # Set up a basic logger for config loading
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Load config from YAML (provide path to your config file here)
    yaml_path = "config.yaml"
    config = load_config(yaml_path, default_config, logger)

    # Create a unique run directory
    run_dir = create_run_directory(base_dir=config["train"]["output_dir"])

    # save the final, merged config to the run directory
    used_config_path = os.path.join(run_dir, "config_used.yaml")
    save_config(config, used_config_path)
    logger.info(f"Saved the used config to {used_config_path}")

    # Extract separate config dictionaries
    train_config = config["train"]
    model_config = config["model"]

    # Start training
    trainer = Trainer(train_config, model_config, run_dir)
    trainer.train()


if __name__ == "__main__":
    main()
