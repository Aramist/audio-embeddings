import torch


class AudioEmbedder(torch.nn.Module):
    @classmethod
    def from_pretrained(cls):
        """Should download/load pretrained weights and return an instance
        of the embedder with those weights loaded.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(
            "Subclasses must implement from_pretrained initializer."
        )

    def __init__(self):
        super(AudioEmbedder, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the embedder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, num_samples).

        Returns:
            torch.Tensor: Output embeddings of shape (batch_size, embedding_dim).
        """
        raise NotImplementedError("Subclasses must implement forward method.")
