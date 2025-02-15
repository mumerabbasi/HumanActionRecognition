import os
import argparse
from src.utils.preprocess_data_utils import (
    organize_actions, stratified_split_dataset
)


def parse_arguments():
    """
    Parse command-line arguments for data preprocessing.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Preprocess dataset \
                                     for training.")

    parser.add_argument(
        "--source_dir", type=str, required=True,
        help="Path to the source directory containing raw data."
    )
    parser.add_argument(
        "--dest_dir", type=str, required=True,
        help="Path to the destination directory for processed data."
    )
    parser.add_argument(
        "--train_pct", type=float, default=0.7,
        help="Fraction of data used for training (default: 0.7)."
    )
    parser.add_argument(
        "--val_pct", type=float, default=0.15,
        help="Fraction of data used for validation (default: 0.15)."
    )
    parser.add_argument(
        "--test_pct", type=float, default=0.15,
        help="Fraction of data used for testing (default: 0.15)."
    )
    parser.add_argument(
        "--config", type=str, default=os.path.join('configs', 'custom.yaml'),
        help="Path to an optional configuration file for other settings."
    )

    return parser.parse_args()


def main():
    """
    Main function to preprocess the dataset by:
      1) Organizing actions into structured folders.
      2) Splitting into train, validation, and test sets.
    """
    args = parse_arguments()

    # Define annotations file path (one level above source_dir)
    action_labels_csv = os.path.join(
        os.path.dirname(args.source_dir), 'annotations', 'action_labels.csv'
    )

    print("Organizing actions into folders...")
    organize_actions(
        src_dir=args.source_dir,
        target_dir=args.dest_dir,
        action_labels_csv=action_labels_csv
    )

    print("Splitting data into train, validation, and test sets...")
    stratified_split_dataset(
        data_dir=args.dest_dir,
        out_dir=args.dest_dir,
        train_pct=args.train_pct,
        val_pct=args.val_pct,
        test_pct=args.test_pct
    )

    print("Preprocessing completed.")


if __name__ == "__main__":
    main()
