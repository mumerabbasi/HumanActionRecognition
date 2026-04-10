import torch.nn as nn


class TemporalTransformerBlock(nn.Module):
    """
    Transformer block for modeling the temporal (sequence length) dimension.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of transformer layers.
    dropout : float, optional
        Dropout probability (default is 0.1).
    """
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1):
        super(TemporalTransformerBlock, self).__init__()

        # Transformer encoder layer that will operate on the temporal dimension
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=2 * embed_dim,
            dropout=dropout,
            batch_first=True
        )

        # Stack multiple transformer encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

    def forward(self, x):
        """
        Forward pass through the temporal transformer block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_views, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, num_views, seq_len, embed_dim).
        """
        batch_size, num_views, seq_len, embed_dim = x.shape

        # Reshape input to combine batch_size and num_views for
        # temporal modeling
        x = x.reshape(batch_size * num_views, seq_len, embed_dim)

        # Apply transformer encoder on the temporal dimension
        x = self.transformer_encoder(x)

        # Reshape back to original dimensions
        x = x.reshape(batch_size, num_views, seq_len, embed_dim)

        return x
