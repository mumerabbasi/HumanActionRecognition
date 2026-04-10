from copy import deepcopy
from datetime import datetime
import os

import torch
import yaml


def create_run_directory(base_dir="output/models"):
    """
    Create a new directory for the current training run based on the timestamp.

    Parameters
    ----------
    base_dir : str
        The base directory where run-specific folders will be created.

    Returns
    -------
    str
        The path to the created run directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_config(config, filepath):
    """
    Save a configuration dictionary to a YAML file.

    Parameters
    ----------
    config : dict
        The configuration dictionary to save.
    filepath : str
        The file path where the configuration should be saved.

    Returns
    -------
    None
        This function does not return anything.
    """
    with open(filepath, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file)


def load_config(yaml_path, default_config, logger):
    """
    Load the configuration from a YAML file and merge it with the default
    configuration.

    Parameters
    ----------
    yaml_path : str
        Path to the YAML configuration file.
    default_config : dict
        Default configuration dictionary with 'model' and 'train' sub-dicts.
    logger : logging.Logger
        Logger instance to log any warnings.

    Returns
    -------
    dict
        Merged configuration dictionary with 'model' and 'train' sub-dicts.
    """
    if not os.path.isfile(yaml_path):
        logger.warning(
            f"YAML configuration file not found at {yaml_path}. "
            "Using default configuration."
        )
        return deepcopy(default_config)

    with open(yaml_path, "r", encoding="utf-8") as file:
        user_config = yaml.safe_load(file) or {}

    if not isinstance(user_config, dict):
        raise ValueError(f"YAML configuration at {yaml_path} must be a map.")

    merged_config = deepcopy(default_config)
    for section, values in user_config.items():
        if (
            section in merged_config
            and isinstance(merged_config[section], dict)
            and isinstance(values, dict)
        ):
            merged_config[section] = {
                **merged_config[section],
                **values,
            }
        else:
            merged_config[section] = values

    return merged_config


def save_checkpoint(filepath, model, optimizer, loss):
    """
    Save a model checkpoint.

    Parameters
    ----------
    filepath : str
        Path to save the checkpoint.
    model : torch.nn.Module
        The model to save.
    optimizer : torch.optim.Optimizer
        The optimizer used during training.
    loss : float
        The validation loss at the time of saving.
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        },
        filepath,
    )
