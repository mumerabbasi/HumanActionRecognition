import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class SpatialFeatureExtractor(nn.Module):
    """
    Spatial feature extractor block using ResNet-50.

    Parameters
    ----------
    pretrained : bool, optional
        Whether to use a ResNet-50 model pretrained on ImageNet
        (default is True).
    """

    def __init__(self, pretrained=True):
        super(SpatialFeatureExtractor, self).__init__()

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        self.resnet = resnet50(weights=weights)
        self.embed_dim = self.resnet.fc.in_features

        # Keep convolutional layers plus average pooling, drop the classifier.
        self.feature_extractor = nn.Sequential(*list(
            self.resnet.children()
        )[:-1])

        # Freeze all layers except the last convolutional block
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.feature_extractor[-2].parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Forward pass through the spatial feature extractor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_views, seq_len, C, H, W).

        Returns
        -------
        torch.Tensor
            Extracted features of shape
            (batch_size, num_views, seq_len, embed_dim).
        """
        batch_size, num_views, seq_len, C, H, W = x.shape

        # Reshape input for processing
        x = x.reshape(batch_size * num_views * seq_len, C, H, W)

        # Extract pooled ResNet-50 features.
        features = self.feature_extractor(x)
        features = features.reshape(features.size(0), -1)

        # Reshape features back to original batch structure
        features = features.reshape(batch_size, num_views, seq_len, -1)

        return features
