import glob
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MultiviewActionDataset(Dataset):
    """
    Custom Dataset for Multiview Action Recognition.

    Parameters
    ----------
    data_dir : str
        Path to the processed dataset directory (e.g., 'data/processed').
    transform : callable, optional
        A function/transform that takes in an image and returns a transformed
        version.
    seq_len : int, optional
        Fixed length for sequences. If None, the original sequence length is
        used (default is None).
    """

    def __init__(self, data_dir, transform=None, seq_len=None):
        self.data_dir = data_dir
        self.transform = transform
        self.seq_len = seq_len
        self.actions = sorted(
            action for action in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, action))
        )
        self.num_classes = len(self.actions)

        # Create a mapping from action labels (strings) to integers
        self.action_to_idx = {action: idx for idx,
                              action in enumerate(self.actions)}
        self.data = self._load_dataset()

    def _load_dataset(self):
        """
        Loads the dataset paths into memory.

        Returns
        -------
        list of tuple
            List of tuples in the format (action, sequence_id, view_paths).
        """
        data = []
        for action in self.actions:
            action_dir = os.path.join(self.data_dir, action)
            sequences = sorted(os.listdir(action_dir))
            for seq in sequences:
                seq_dir = os.path.join(action_dir, seq)
                views = sorted(
                    view for view in glob.glob(os.path.join(seq_dir, "*"))
                    if os.path.isdir(view)
                )
                data.append((action, seq, views))
        return data

    def __len__(self):
        """
        Returns the total number of sequences in the dataset.

        Returns
        -------
        int
            The total number of sequences.
        """
        return len(self.data)

    def _load_images(self, view_paths):
        """
        Loads images for all views of a single sequence.

        Parameters
        ----------
        view_paths : list of str
            List of paths for each view in the sequence.

        Returns
        -------
        torch.Tensor
            Tensor of shape (num_views, seq_len, C, H, W), where num_views is
            the number of views.
        """
        images_per_view = []
        for view_path in view_paths:
            frames = sorted(glob.glob(os.path.join(view_path, "*.jpg")))
            if self.seq_len and self.seq_len <= len(frames):
                frames = frames[:self.seq_len]
            images = []
            for frame in frames:
                with Image.open(frame) as image:
                    image = image.convert("RGB")
                    if self.transform:
                        images.append(self.transform(image))
                    else:
                        images.append(transforms.ToTensor()(image))
            images_per_view.append(torch.stack(images))
        return torch.stack(images_per_view, dim=0)

    def __getitem__(self, idx):
        """
        Fetches the data at the specified index.

        Parameters
        ----------
        idx : int
            Index for the item to fetch.

        Returns
        -------
        tuple
            A tuple containing:
                - torch.Tensor: View frames of shape
                (num_views, seq_len, C, H, W).
                - torch.Tensor: Corresponding action index.
        """
        action, _seq, view_paths = self.data[idx]
        views = self._load_images(view_paths)
        action_idx = torch.tensor(self.action_to_idx[action])
        return views, action_idx
