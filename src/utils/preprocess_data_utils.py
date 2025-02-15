import os
import shutil
import logging

import pandas as pd
from sklearn.model_selection import train_test_split


logger = logging.getLogger(__name__)


def organize_actions(
    src_dir: str,
    target_dir: str,
    action_labels_csv: str,
    stride: int = 24,
    max_frame_diff: int = 48,
    num_views: int = 8
) -> None:
    """
    Organize actions into folders structured as ``/action_x/seq_x/view_x`` by
    reading a CSV file and building multiple sequences (subgroups of frames).

    Parameters
    ----------
    src_dir : str
        The source directory containing the raw data.
    target_dir : str
        The directory where organized action folders will be stored.
    action_labels_csv : str
        Path to the CSV file containing action labels and frame information.
    stride : int, optional
        Number of frames to stride between consecutive subgroups 
        (default is 24).
    max_frame_diff : int, optional
        Maximum number of frames to consider in a subgroup (default is 48).
    num_views : int, optional
        The number of views (camera angles) per action (default is 8).

    Returns
    -------
    None
        This function does not return anything. It creates directories and
        copies frames into organized folders.
    """
    df = pd.read_csv(action_labels_csv)

    # Dictionary to keep track of sequence indices for each action label
    action_seq_count = {}

    for index, row in df.iterrows():
        action_label = row["action_label"].replace(" ", "_")

        if action_label not in action_seq_count:
            action_seq_count[action_label] = 0

        total_frames = row["frame_diff"]
        start_frame = row["start frame"]

        # Calculate number of subgroups based on stride
        num_subgroups = (total_frames - stride) // stride + 1
        total_frames = min(total_frames, max_frame_diff)

        if num_subgroups <= 0:
            logger.warning(
                "Skipping action due to insufficient frames for row %s", index
            )
            continue

        for subgroup_index in range(num_subgroups):
            action_seq_count[action_label] += 1
            seq_folder = f"seq_{action_seq_count[action_label]}"

            for view in range(1, num_views + 1):
                act_folder = os.path.join(
                    target_dir, action_label, seq_folder, f"view_{view}"
                )

                subgroup_start = start_frame + subgroup_index * stride
                subgroup_end = subgroup_start + max_frame_diff

                # Ensure end does not exceed the total frames for this action
                subgroup_end = min(subgroup_end,
                                   start_frame + row["frame_diff"])

                if (subgroup_end - subgroup_start) < stride:
                    continue

                os.makedirs(act_folder, exist_ok=True)

                src_folder = os.path.join(
                    src_dir,
                    str(row["sequence_number"]),
                    f"{row['sequence_number']}_view_{view}"
                )

                for frame in range(subgroup_start, subgroup_end + 1):
                    src_file = os.path.join(src_folder, f"left{frame:04d}.jpg")
                    tgt_file = os.path.join(act_folder, f"left{frame:04d}.jpg")

                    if os.path.exists(src_file):
                        shutil.copy(src_file, tgt_file)
                    else:
                        logger.warning("File does not exist: %s", src_file)

    logger.info("Organized actions into separate folders.")


def stratified_split_dataset(
    data_dir: str,
    out_dir: str,
    train_pct: float = 0.8,
    val_pct: float = 0.1,
    test_pct: float = 0.1
) -> None:
    """
    Perform a stratified split of a multiview dataset into train, validation,
    and test sets, while preserving directory structure.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the dataset with structure
        ``data/action_label/seq_x/view_x``.
    out_dir : str
        Path to the output directory where ``train``, ``val``, and ``test``
        directories will be created.
    train_pct : float, optional
        Percentage of data allocated to the training set (default is 0.8).
    val_pct : float, optional
        Percentage of data allocated to the validation set (default is 0.1).
    test_pct : float, optional
        Percentage of data allocated to the test set (default is 0.1).

    Raises
    ------
    AssertionError
        If the sum of ``train_pct``, ``val_pct``, and ``test_pct`` is not
        equal to 1.

    Returns
    -------
    None
        The function moves directories to their respective ``train``, ``val``,
        or ``test`` folders and removes empty directories from the original
        location.
    """
    assert (
        abs(train_pct + val_pct + test_pct - 1.0) < 1e-6
    ), "Train, Val, and Test percentages must sum to 1."

    actions = [
        action for action in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, action))
    ]
    action_to_seqs = {}

    for action in actions:
        action_path = os.path.join(data_dir, action)
        seqs = [
            seq for seq in os.listdir(action_path)
            if os.path.isdir(os.path.join(action_path, seq))
        ]
        action_to_seqs[action] = seqs

    train_dir = os.path.join(out_dir, "train")
    val_dir = os.path.join(out_dir, "val")
    test_dir = os.path.join(out_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # For each action, perform a stratified split on sequences
    for action, seqs in action_to_seqs.items():
        train_seqs, temp_seqs = train_test_split(
            seqs,
            test_size=(val_pct + test_pct),
            random_state=42
        )
        val_seqs, test_seqs = train_test_split(
            temp_seqs,
            test_size=(test_pct / (val_pct + test_pct)),
            random_state=42
        )

        for seq in train_seqs:
            move_sequence(data_dir, train_dir, action, seq)
        for seq in val_seqs:
            move_sequence(data_dir, val_dir, action, seq)
        for seq in test_seqs:
            move_sequence(data_dir, test_dir, action, seq)

    remove_empty_dirs(data_dir)
    logger.info("Data successfully split into train, val, and test sets.")


def move_sequence(
    data_dir: str,
    target_dir: str,
    action: str,
    seq: str
) -> None:
    """
    Move a sequence directory from the data directory to the target directory,
    preserving the action and sequence structure.

    Parameters
    ----------
    data_dir : str
        Path to the original data directory.
    target_dir : str
        Path to the target directory where data should be moved.
    action : str
        Action label (sub-directory name).
    seq : str
        Sequence identifier (sub-directory name).

    Returns
    -------
    None
        This function moves a directory from one location to another.
    """
    action_dir = os.path.join(target_dir, action)
    os.makedirs(action_dir, exist_ok=True)

    seq_dir = os.path.join(data_dir, action, seq)
    target_seq_dir = os.path.join(action_dir, seq)

    shutil.move(seq_dir, target_seq_dir)
    logger.debug(
        "Moved sequence '%s' of action '%s' to %s", seq,
        action, target_seq_dir
    )


def remove_empty_dirs(root_dir: str) -> None:
    """
    Recursively remove empty directories from a root directory.

    Parameters
    ----------
    root_dir : str
        Path to the root directory to clean up.

    Returns
    -------
    None
        This function removes empty directories from the file system in-place.
    """
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        if not dirnames and not filenames:
            os.rmdir(dirpath)
            logger.debug("Removed empty directory %s", dirpath)
