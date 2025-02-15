import torchvision.transforms as T


def get_train_transforms():
    """
    Returns the set of transformations to be applied during training.

    The transformations include resizing, normalization, and spatial
    augmentations such as random horizontal flipping and random rotation.

    Returns
    -------
    torchvision.transforms.Compose
        A composition of image transformations for training.
    """
    transform = T.Compose([
        T.Resize(size=(224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=(-10, 10)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    return transform


def get_val_transforms():
    """
    Returns the set of transformations to be applied during validation and
    testing.

    The transformations include resizing and normalization, without data
    augmentation.

    Returns
    -------
    torchvision.transforms.Compose
        A composition of image transformations for validation/testing.
    """
    transform = T.Compose([
        T.Resize(size=(224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    return transform