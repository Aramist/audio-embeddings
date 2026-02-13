import torch
from torch import nn


class AudioTransformation(nn.Module):
    """
    Abstract class for audio transformations.
    """

    def __init__(self):
        super(AudioTransformation, self).__init__()

    def apply(
        self, audio: torch.Tensor, sample_rate: float
    ) -> tuple[torch.Tensor, float]:
        """
        Apply the transformation to the input waveform.

        Args:
            audio (torch.Tensor): The 2D audio waveform tensor.
                audio shape: (batch_size, num_samples)
            sample_rate (float): The sample rate of the audio waveform.

        Returns:
            tuple[torch.Tensor, float]: Transformed audio waveform tensor and its sample rate. Output audio will
                have expanded shape: (batch_size, augmentation_dim, num_samples)
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def forward(self, audio: tuple[torch.Tensor, float]) -> tuple[torch.Tensor, float]:
        """
        Apply the transformation to the input waveform.

        Args:
            audio (tuple[torch.Tensor, float]): A tuple containing the audio waveform tensor and its sample rate.
                audio shape: (*batch_dims, num_samples)
        Returns:
            tuple[torch.Tensor, float]: Transformed audio waveform tensor and its sample rate. Output audio will
                have expanded shape: (*batch_dims, augmentation_dim, num_samples)
        """
        if not isinstance(audio, tuple) or len(audio) != 2:
            raise ValueError("Input must be a tuple of (tensor, sample_rate).")

        batch_dims = audio[0].shape[:-1]
        audio = (
            audio[0].reshape(-1, audio[0].shape[-1]),
            audio[1],
        )  # Reshape to (batch_size, num_samples)
        aug_audio, sr = self.apply(
            *audio
        )  # (batch_size, augmentation_dim, num_samples), sample_rate
        aug_audio = aug_audio.reshape(*batch_dims, *aug_audio.shape[1:])
        return aug_audio, sr
