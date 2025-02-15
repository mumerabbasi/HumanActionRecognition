def calculate_accuracy(correct, total):
    """
    Calculate accuracy.

    Args:
        correct (int): Number of correct predictions.
        total (int): Total number of predictions.

    Returns:
        float: Accuracy in percentage.
    """
    return 100 * correct / total if total > 0 else 0
