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

    def __init__(self, auto_convert_sample_rate: bool = False):
        super(AudioEmbedder, self).__init__()
        self.auto_convert_sample_rate = auto_convert_sample_rate

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
            audio (tuple[torch.Tensor, float]): A tuple containing the input tensor of shape (*batch_dims, num_channels, num_samples)
            and the sample rate as a float.
        Returns:
            torch.Tensor: Output embeddings of shape (batch_size, num_channels, embedding_dim).
        """

        if not isinstance(audio, tuple) or len(audio) != 2:
            raise ValueError("Input audio must be a tuple of (tensor, sample_rate).")

        if self.expected_sample_rate is None:
            raise ValueError("expected_sample_rate must be defined by subclass.")

        signal = audio[0]
        batch_dims = ()

        if audio[1] != self.expected_sample_rate:
            if not self.auto_convert_sample_rate:
                raise ValueError(
                    f"Input sample rate {audio[1]} does not match expected sample rate {self.expected_sample_rate} and auto_convert_sample_rate is False."
                )
            # resample audio
            orig_device = signal.device
            resampled_audio = resample(
                signal.cpu().numpy(),
                orig_sr=audio[1],
                target_sr=self.expected_sample_rate,
            )
            resampled_audio = torch.from_numpy(resampled_audio).float().to(orig_device)
            signal = resampled_audio
        else:
            signal = signal.float()

        if signal.ndim == 2:
            # Missing batch dimension
            signal = signal.unsqueeze(0)
        elif signal.ndim == 1:
            # Missing batch and channel dimensions
            signal = signal.unsqueeze(0).unsqueeze(0)
        elif signal.ndim > 3:
            batch_dims = signal.shape[:-2]
            signal = signal.reshape(-1, signal.shape[-2], signal.shape[-1])

        with torch.no_grad():
            embedding = self.embed(signal)
        if batch_dims:
            embedding = embedding.reshape(*batch_dims, *embedding.shape[1:])

        return embedding
