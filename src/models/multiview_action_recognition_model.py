import torch.nn as nn
from src.models.spatial_feature_extractor import SpatialFeatureExtractor
from src.models.attention_views import MultiHeadAttentionOnViews
from src.models.transformer_encoder_temporal import TemporalTransformerBlock


class MultiviewActionRecognitionModel(nn.Module):
    """
    Multiview action recognition model using ResNet-50 spatial features,
    multi-head attention on views, and a Transformer encoder for temporal
    modeling.

    Parameters
    ----------
    num_heads : int, optional
        Number of attention heads (default is 4).
    pretrained_spatial_feature_extractor : bool, optional
        Whether to use a pretrained spatial feature extractor
        (default is True).
    num_transformer_layers : int, optional
        Number of Transformer layers (default is 2).
    num_classes : int, optional
        Number of output classes (default is 6).
    """
    def __init__(
        self,
        num_heads=4,
        pretrained_spatial_feature_extractor=True,
        num_transformer_layers=2,
        num_classes=6,
    ):
        super(MultiviewActionRecognitionModel, self).__init__()

        # Feature extractor
        self.spatial_feature_extractor = SpatialFeatureExtractor(
            pretrained_spatial_feature_extractor
        )

        # Embedding dimension of the feature extractor
        self.embed_dim = self.spatial_feature_extractor.embed_dim

        # Multi-Head Attention on views
        self.attention_on_views = MultiHeadAttentionOnViews(
            self.embed_dim, num_heads
        )

        # Transformer for temporal modeling
        self.temporal_model = TemporalTransformerBlock(
            self.embed_dim, num_heads, num_transformer_layers
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, num_classes)
        )

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_views, seq_len, C, H, W).

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, num_classes).
        """
        # Extract spatial features for each frame
        features = self.spatial_feature_extractor(x)

        # Apply multi-head attention on views
        attn_output = self.attention_on_views(features)

        # Apply temporal Transformer on sequence
        temporal_transformer_output = self.temporal_model(attn_output)

        # Pool over the temporal dimension (seq_len)
        pooled_features = temporal_transformer_output.mean(dim=2)

        # Pool over the views dimension (num_views)
        final_features = pooled_features.mean(dim=1)

        # Classification
        logits = self.classifier(final_features)

        return logits
