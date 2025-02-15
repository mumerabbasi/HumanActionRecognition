import logging
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

from src.data.dataset import MultiviewActionDataset
from src.models.multiview_action_recognition_model import (
    MultiviewActionRecognitionModel,
)
from utils.logger import setup_logger
from utils.helper import (
    create_run_directory,
    load_config,
    save_config,
)


class Tester:
    """
    Tester class for the human action segmentation (or recognition) model.

    This class handles the loading of a trained model and performing
    inference on a test dataset to evaluate metrics such as F1-score,
    confusion matrix, and accuracy.

    Attributes
    ----------
    test_config : dict
        Configuration for the testing process (batch size, logging, etc.).
    model_config : dict
        Configuration for the model (same as used in training).
    run_dir : str
        Directory for logs/results in the current test run.
    device : torch.device
        Device to run the inference on (CPU or CUDA).
    logger : logging.Logger
        Logger instance for logging information.
    model : nn.Module
        Loaded model for inference.
    use_wandb : bool
        Indicates whether to log metrics to Weights & Biases.
    test_loader : torch.utils.data.DataLoader
        DataLoader for the test dataset.
    """

    def __init__(
            self, test_config: dict, model_config: dict, run_dir: str
    ) -> None:
        """
        Initialize the Tester class with the provided configuration.

        Parameters
        ----------
        test_config : dict
            Testing-related configuration dictionary.
        model_config : dict
            Model-related configuration dictionary.
        run_dir : str
            Directory for the current test run (for logs/results).
        """
        self.test_config = test_config
        self.model_config = model_config
        self.run_dir = run_dir

        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")

        # Setup logger to log into run_dir/test.log
        self.logger = setup_logger(
            "test_log", os.path.join(run_dir, "test.log")
        )

        # Check if wandb is enabled
        self.use_wandb = (
            str(self.test_config.get("wandb", "no")).lower() == "yes"
        )
        if self.use_wandb:
            wandb.init(project="multiview-action-recognition", config={
                "test": self.test_config,
                "model": self.model_config
            })
            self.logger.info("WandB logging is enabled for testing.")
        else:
            self.logger.info("WandB logging is disabled for testing.")

        # Initialize the model
        self.model = MultiviewActionRecognitionModel(
            num_heads=self.model_config["num_heads"],
            num_transformer_layers=self.model_config["num_transformer_layers"],
            num_classes=self.model_config["num_classes"],
        ).to(self.device)

        # Create the test dataloader
        self.test_loader = self.get_test_dataloader()

    def get_test_dataloader(self) -> DataLoader:
        """
        Initialize the test data loader.

        Returns
        -------
        DataLoader
            DataLoader for the test dataset.
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        seq_len = self.test_config.get("seq_len", 50)

        test_dataset = MultiviewActionDataset(
            data_dir="data/processed/test",
            transform=transform,
            seq_len=seq_len
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.test_config["batch_size"],
            shuffle=False,
            num_workers=4
        )

        return test_loader

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load the model weights from the specified checkpoint.

        Parameters
        ----------
        checkpoint_path : str
            Path to the model checkpoint file.
        """
        self.logger.info(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.logger.info("Checkpoint loaded successfully.")

    def test(self) -> None:
        """
        Run inference on the test dataset and log metrics:
        F1-score, confusion matrix, and accuracy.
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        # Loss function is optional for test—only for an average loss
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0

        with torch.no_grad():
            for views, actions in self.test_loader:
                views, actions = views.to(self.device), actions.to(self.device)

                outputs = self.model(views)
                loss = criterion(outputs, actions)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(actions.cpu().numpy().tolist())

        # Compute metrics
        f1 = f1_score(all_labels, all_preds, average="weighted")
        acc = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        avg_loss = total_loss / len(self.test_loader)

        # Log metrics
        self.logger.info(f"Test Loss: {avg_loss:.4f}")
        self.logger.info(f"Test Accuracy: {acc*100:.2f}%")
        self.logger.info(f"Test F1-score (weighted): {f1:.4f}")
        self.logger.info(f"Confusion Matrix:\n{cm}")

        # Log metrics to wandb if enabled
        if self.use_wandb:
            wandb.log({
                "test_loss": avg_loss,
                "test_accuracy": acc,
                "test_f1_score": f1,
                # TODO: log the confusion matrix as an image/table
                # or use wandb.plot.confusion_matrix for a more
                # visual approach.
            })


def main() -> None:
    """
    Main entry point for testing the human action segmentation
    (or recognition) model.
    """
    # Default configuration for testing
    default_config = {
        "model": {
            "num_heads": 4,
            "num_transformer_layers": 2,
            "num_classes": 10,
        },
        "test": {
            "batch_size": 8,
            "seq_len": 50,
            "wandb": "no",
            "output_dir": "output/test_results",
        }
    }

    # Set up a basic logger for config loading
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Load config from YAML
    yaml_path = "config_test.yaml"
    config = load_config(yaml_path, default_config, logger)

    # Create a unique run directory for testing
    run_dir = create_run_directory(base_dir=config["test"]["output_dir"])

    # Save the final, merged config to the run directory
    used_config_path = os.path.join(run_dir, "config_used_test.yaml")
    save_config(config, used_config_path)
    logger.info(f"Saved the used config to {used_config_path}")

    # Extract separate config dictionaries
    test_config = config["test"]
    model_config = config["model"]

    # Initialize tester
    tester = Tester(test_config, model_config, run_dir)

    # Load the best checkpoint
    checkpoint_to_load = "best_model_epoch_10.pth"
    checkpoint_path = os.path.join("output", "models", checkpoint_to_load)
    tester.load_checkpoint(checkpoint_path)

    # Start testing
    tester.test()


if __name__ == "__main__":
    main()
