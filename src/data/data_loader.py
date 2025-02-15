import os

from torch.utils.data import DataLoader
from src.data.dataset import MultiviewActionDataset


def create_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    seq_len: int = None,
    num_workers: int = 4,
    transform=None
):
    """
    Create PyTorch DataLoaders for training, validation, and testing sets.

    Parameters
    ----------
    data_dir : str
        Path to the processed dataset directory containing 'train', 'val',
        and 'test' sub-directories.
    batch_size : int, optional
        Batch size for the DataLoader (default is 8).
    seq_len : int, optional
        Fixed length for sequences. If None, the original sequence length
        is used (default is None).
    num_workers : int, optional
        Number of workers for data loading (default is 4).
    transform : callable, optional
        A function or transform to apply to the images (default is None).

    Returns
    -------
    tuple
        A tuple of (train_loader, val_loader, test_loader), where each element
        is a PyTorch DataLoader for the respective dataset split.
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    train_dataset = MultiviewActionDataset(
        data_dir=train_dir,
        transform=transform,
        seq_len=seq_len
    )
    val_dataset = MultiviewActionDataset(
        data_dir=val_dir,
        transform=transform,
        seq_len=seq_len
    )
    test_dataset = MultiviewActionDataset(
        data_dir=test_dir,
        transform=transform,
        seq_len=seq_len
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
