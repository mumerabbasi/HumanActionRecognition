import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class SpatialFeatureExtractor(nn.Module):
    """
    Spatial Feature Extractor block using EfficientNetV2-S.

    Parameters
    ----------
    pretrained : bool, optional
        Whether to use a pretrained EfficientNetV2-S model (default is True).
    """
    def __init__(self, pretrained=True):
        super(SpatialFeatureExtractor, self).__init__()

        # Load EfficientNetV2-S pre-trained on ImageNet
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        self.efficient_net = efficientnet_v2_s(weights=weights)

        # Remove the classification head (retain feature extractor)
        self.feature_extractor = nn.Sequential(*list(
            self.efficient_net.children())[:-1]
        )

        # Freeze all layers except the last convolutional block
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.feature_extractor[-1].parameters():
            param.requires_grad = True

        # Determine embedding dimension dynamically using a dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            dummy_output = self.feature_extractor(dummy_input)
        self.embed_dim = dummy_output.shape[1]

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
        x = x.view(batch_size * num_views * seq_len, C, H, W)

        # Extract features using EfficientNetV2-S
        features = self.feature_extractor(x)
        # Remove spatial dimensions
        features = features.squeeze(-1).squeeze(-1)

        # Reshape features back to original batch structure
        features = features.view(batch_size, num_views, seq_len, -1)

        return features
