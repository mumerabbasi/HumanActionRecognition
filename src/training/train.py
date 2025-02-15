import os
import torch
import wandb
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from src.models.multiview_action_recognition_model import MultiviewActionRecognitionModel
from src.datasets.multiview_action_dataset import MultiviewActionDataset
from helpers import save_checkpoint, load_checkpoint, get_lr
from loggers import setup_logger
from metrics import calculate_accuracy


class Trainer:
    def __init__(self, config):
        """
        Initialize the Trainer class with the provided configuration.

        Args:
            config (dict): Configuration for the training process.
        """
        self.config = config
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
            )
        self.logger = setup_logger('training_log', 'training.log')

        # Initialize wandb for logging
        wandb.init(project="multiview-action-recognition", config=self.config)

        # Initialize the model, optimizer, criterion, and dataloaders
        self.model = MultiviewActionRecognitionModel(
            num_heads=self.config['num_heads'],
            num_transformer_layers=self.config['num_transformer_layers'],
            num_classes=self.config['num_classes']
        ).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.criterion = nn.CrossEntropyLoss()

        self.train_loader, self.val_loader = self.get_dataloaders()

        # Create checkpoint directory
        if not os.path.exists(self.config["checkpoint_dir"]):
            os.makedirs(self.config["checkpoint_dir"])

    def get_dataloaders(self):
        """
        Initialize the training and validation data loaders.

        Returns:
            tuple: Training and validation dataloaders.
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        train_dataset = MultiviewActionDataset(
            data_dir='data/processed/train',
            transform=transform,
            seq_len=self.config['seq_len']
        )
        val_dataset = MultiviewActionDataset(
            data_dir='data/processed/val',
            transform=transform,
            seq_len=self.config['seq_len']
        )

        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], shuffle=False, num_workers=4)

        return train_loader, val_loader

    def train_one_epoch(self, epoch):
        """
        Train the model for one epoch.

        Args:
            epoch (int): Current epoch number.

        Returns:
            float: Training loss for the epoch.
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

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Log loss and accuracy
            _, predicted = torch.max(outputs, 1)
            total_predictions += actions.size(0)
            correct_predictions += (predicted == actions).sum().item()
            running_loss += loss.item()

            if batch_idx % self.config['log_interval'] == 0:
                accuracy = calculate_accuracy(correct_predictions, total_predictions)
                self.logger.info(f"Epoch [{epoch+1}/{self.config['num_epochs']}], "
                                 f"Batch [{batch_idx}/{len(self.train_loader)}], "
                                 f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

                # Log to wandb
                wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy, "learning_rate": get_lr(self.optimizer)})

        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = 100 * correct_predictions / total_predictions

        self.logger.info(f"Epoch [{epoch+1}/{self.config['num_epochs']}], "
                         f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%")

        return epoch_loss

    def validate(self):
        """
        Validate the model on the validation set.

        Returns:
            tuple: Validation loss and accuracy.
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
        val_accuracy = 100 * correct_predictions / total_predictions

        self.logger.info(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')
        return val_loss, val_accuracy

    def train(self):
        """
        Full training loop with validation and checkpointing.
        """
        best_val_loss = float('inf')

        for epoch in range(self.config['num_epochs']):
            # Train for one epoch
            train_loss = self.train_one_epoch(epoch)

            # Validate the model
            val_loss, val_accuracy = self.validate()

            # Save checkpoint if validation loss improves
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(self.config["checkpoint_dir"], f"best_model_epoch_{epoch+1}.pth")
                save_checkpoint(checkpoint_path, self.model, self.optimizer, best_val_loss)
                self.logger.info(f"Checkpoint saved at {checkpoint_path}")

            # Log validation metrics to wandb
            wandb.log({"val_loss": val_loss, "val_accuracy": val_accuracy})

            # Early stopping or other logic can be added here

# Configuration and Hyperparameters
config = {
    "batch_size": 8,
    "num_epochs": 50,
    "learning_rate": 1e-4,
    "seq_len": 50,
    "num_classes": 10,
    "num_heads": 4,
    "num_transformer_layers": 2,
    "checkpoint_dir": "checkpoints",
    "log_interval": 10,
}

if __name__ == "__main__":
    trainer = Trainer(config)
    trainer.train()
