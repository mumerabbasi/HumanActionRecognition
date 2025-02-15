import torch.nn as nn


class MultiHeadAttentionOnViews(nn.Module):
    """
    Multi-head attention block applied to views.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    num_heads : int, optional
        Number of attention heads (default is 4).
    """
    def __init__(self, embed_dim, num_heads=4):
        super(MultiHeadAttentionOnViews, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            batch_first=True, dropout=0.1
        )

    def forward(self, x):
        """
        Forward pass through the multi-head attention mechanism over views.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, num_views, seq_len, embed_dim).

        Returns
        -------
        torch.Tensor
            Output tensor with attention applied over views, of shape
            (batch_size, num_views, seq_len, embed_dim).
        """
        batch_size, num_views, seq_len, embed_dim = x.shape

        # Reshape input for attention over views
        x = x.permute(0, 2, 1, 3).contiguous().view(
            batch_size * seq_len,
            num_views, embed_dim
        )

        # Apply multi-head attention across the views
        attn_output, _ = self.multihead_attn(x, x, x)

        # Reshape back to original shape
        attn_output = attn_output.view(
            batch_size, seq_len, num_views, embed_dim
        )
        attn_output = attn_output.permute(0, 2, 1, 3)

        return attn_output
