import argparse
import logging
import os

import torch
import torch.nn as nn
import wandb
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.data.dataset import MultiviewActionDataset
from src.data.transforms import get_val_transforms
from src.models.multiview_action_recognition_model import (
    MultiviewActionRecognitionModel,
)
from src.utils.helper import (
    create_run_directory,
    load_config,
    save_config,
)
from src.utils.logger import setup_logger


class Evaluator:
    """
    Evaluator for the multiview action recognition model.

    This class handles the loading of a trained model and performing
    inference on an evaluation split to compute metrics such as macro F1,
    confusion matrix, and accuracy.

    Attributes
    ----------
    eval_config : dict
        Configuration for evaluation (batch size, logging, etc.).
    model_config : dict
        Configuration for the model (same as used in training).
    run_dir : str
        Directory for logs/results in the current evaluation run.
    device : torch.device
        Device to run the inference on (CPU or CUDA).
    logger : logging.Logger
        Logger instance for logging information.
    model : nn.Module
        Loaded model for inference.
    use_wandb : bool
        Indicates whether to log metrics to Weights & Biases.
    eval_loader : torch.utils.data.DataLoader
        DataLoader for the evaluation dataset.
    """

    def __init__(
            self, eval_config: dict, model_config: dict, run_dir: str
    ) -> None:
        """
        Initialize the Evaluator class with the provided configuration.

        Parameters
        ----------
        eval_config : dict
            Evaluation-related configuration dictionary.
        model_config : dict
            Model-related configuration dictionary.
        run_dir : str
            Directory for the current evaluation run (for logs/results).
        """
        self.eval_config = eval_config
        self.model_config = model_config
        self.run_dir = run_dir

        self.device = torch.device("cuda" if torch.cuda.is_available()
                                   else "cpu")

        # Setup logger to log into run_dir/evaluate.log
        self.logger = setup_logger(
            "evaluate_log", os.path.join(run_dir, "evaluate.log")
        )

        # Check if wandb is enabled
        self.use_wandb = (
            str(self.eval_config.get("wandb", "no")).lower() == "yes"
        )
        if self.use_wandb:
            wandb.init(project="multiview-action-recognition", config={
                "evaluate": self.eval_config,
                "model": self.model_config
            })
            self.logger.info("WandB logging is enabled for evaluation.")
        else:
            self.logger.info("WandB logging is disabled for evaluation.")

        # Initialize the model
        self.model = MultiviewActionRecognitionModel(
            num_heads=self.model_config["num_heads"],
            pretrained_spatial_feature_extractor=self.model_config.get(
                "pretrained_spatial_feature_extractor", True
            ),
            num_transformer_layers=self.model_config["num_transformer_layers"],
            num_classes=self.model_config["num_classes"],
        ).to(self.device)

        self.eval_loader = self.get_eval_dataloader()

    def get_eval_dataloader(self) -> DataLoader:
        """
        Initialize the evaluation data loader.

        Returns
        -------
        DataLoader
            DataLoader for the evaluation dataset.
        """
        seq_len = self.eval_config.get("seq_len", 50)
        data_dir = self.eval_config.get("data_dir", "data/processed")
        split = self.eval_config.get("split", "test")
        num_workers = self.eval_config.get("num_workers", 4)

        eval_dataset = MultiviewActionDataset(
            data_dir=os.path.join(data_dir, split),
            transform=get_val_transforms(),
            seq_len=seq_len
        )

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.eval_config["batch_size"],
            shuffle=False,
            num_workers=num_workers
        )

        return eval_loader

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

    def evaluate(self) -> None:
        """
        Run inference and log macro F1, confusion matrix, and accuracy.
        """
        self.model.eval()
        all_preds = []
        all_labels = []

        # Loss is optional during evaluation, but useful for run summaries.
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0

        with torch.no_grad():
            for views, actions in self.eval_loader:
                views, actions = views.to(self.device), actions.to(self.device)

                outputs = self.model(views)
                loss = criterion(outputs, actions)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy().tolist())
                all_labels.extend(actions.cpu().numpy().tolist())

        # Compute metrics
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        acc = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        avg_loss = total_loss / len(self.eval_loader)

        # Log metrics
        self.logger.info(f"Evaluation Loss: {avg_loss:.4f}")
        self.logger.info(f"Evaluation Accuracy: {acc*100:.2f}%")
        self.logger.info(f"Evaluation Macro F1-score: {f1:.4f}")
        self.logger.info(f"Confusion Matrix:\n{cm}")

        # Log metrics to wandb if enabled
        if self.use_wandb:
            wandb.log({
                "eval_loss": avg_loss,
                "eval_accuracy": acc,
                "eval_macro_f1_score": f1,
            })


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate the multiview action recognition model."
    )
    parser.add_argument(
        "--config",
        default=os.path.join("configs", "default.yaml"),
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the checkpoint to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for evaluating the multiview action recognition model.
    """
    args = parse_args()

    # Default configuration for evaluation
    default_config = {
        "model": {
            "num_heads": 4,
            "pretrained_spatial_feature_extractor": True,
            "num_transformer_layers": 2,
            "num_classes": 6,
        },
        "evaluate": {
            "batch_size": 8,
            "seq_len": 50,
            "data_dir": "data/processed",
            "split": "test",
            "num_workers": 4,
            "wandb": "no",
            "output_dir": "output/evaluation_results",
        }
    }

    # Set up a basic logger for config loading
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    # Load config from YAML
    config = load_config(args.config, default_config, logger)

    # Create a unique run directory for evaluation
    run_dir = create_run_directory(base_dir=config["evaluate"]["output_dir"])

    # Save the final, merged config to the run directory
    used_config_path = os.path.join(run_dir, "config_used_evaluate.yaml")
    save_config(config, used_config_path)
    logger.info(f"Saved the used config to {used_config_path}")

    # Extract separate config dictionaries
    eval_config = config["evaluate"]
    model_config = config["model"]

    evaluator = Evaluator(eval_config, model_config, run_dir)

    evaluator.load_checkpoint(args.checkpoint)

    evaluator.evaluate()


if __name__ == "__main__":
    main()
