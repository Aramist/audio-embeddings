import torch
from librosa import resample


class AudioEmbedder(torch.nn.Module):
    expected_sample_rate: float | None = None
    embedding_dim: int | None = None

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

    def embed(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute embeddings for the given audio input.

        Args:
            audio (torch.Tensor): Input audio tensor of shape (batch_size, num_channels, num_samples).
        Returns:
            torch.Tensor: Output embeddings of shape (batch_size, embedding_dim).
        """
        raise NotImplementedError("Subclasses must implement embed method.")

    def forward(self, audio: tuple[torch.Tensor, float]) -> torch.Tensor:
        """Forward pass through the embedder.

        Args:
            audio (tuple[torch.Tensor, float]): A tuple containing the input tensor of shape (batch_size, num_channels, num_samples)
            and the sample rate as a float.
        Returns:
            torch.Tensor: Output embeddings of shape (batch_size, embedding_dim).
        """

        if type(audio) != tuple or len(audio) != 2:
            raise ValueError("Input audio must be a tuple of (tensor, sample_rate).")

        if self.expected_sample_rate is None:
            raise ValueError("expected_sample_rate must be defined by subclass.")

        if audio[1] != self.expected_sample_rate:
            # resample audio
            resampled_audio = resample(
                audio[0].numpy(), orig_sr=audio[1], target_sr=self.expected_sample_rate
            )
            resampled_audio = (
                torch.from_numpy(resampled_audio).to(audio[0].dtype).to(audio[0].device)
            )
            return self.embed(resampled_audio)
        else:
            return self.embed(audio[0])
