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
            audio (torch.Tensor): The audio waveform tensor.
                audio shape: (*batch_dim, num_channels, num_samples)
            sample_rate (float): The sample rate of the audio waveform.

        Returns:
            tuple[torch.Tensor, float]: Transformed audio waveform tensor and its sample rate. Output audio will
                have expanded shape: (*batch_dim, augmentation_dim, num_channels, num_samples)
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def forward(self, audio: tuple[torch.Tensor, float]) -> tuple[torch.Tensor, float]:
        """
        Apply the transformation to the input waveform.

        Args:
            audio (tuple[torch.Tensor, float]): A tuple containing the audio waveform tensor and its sample rate.
                audio shape: (*batch_dim, num_channels, num_samples)
        Returns:
            tuple[torch.Tensor, float]: Transformed audio waveform tensor and its sample rate. Output audio will
                have expanded shape: (*batch_dim, augmentation_dim, num_channels, num_samples)
        """
        if len(audio) != 2:
            raise ValueError("Input must be a tuple of (audio_tensor, sample_rate).")
        if audio[0].ndim == 2:
            # Missing batch dimension
            audio = (audio[0].unsqueeze(0), audio[1])  # Add batch dimension
        elif audio[0].ndim == 1:
            # Missing batch and channel dimensions
            audio = (
                audio[0].unsqueeze(0).unsqueeze(0),
                audio[1],
            )  # Add batch and channel dimensions
        return self.apply(*audio)
