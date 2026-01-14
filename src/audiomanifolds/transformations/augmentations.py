import numpy as np
import torch

from .base import AudioTransformation


class Gain(AudioTransformation):
    """Applies a batch of gains to the input audio signal."""

    def __init__(self, gains: list[float] | np.ndarray | torch.Tensor):
        """
        Args:
            gains (list[float] | np.ndarray | torch.Tensor): List of gain values in decibels to apply to each channel of audio.
        """
        super(Gain, self).__init__()
        if not isinstance(gains, torch.Tensor):
            gains = torch.tensor(gains)
        self.gains = gains

    def apply(self, audio: torch.Tensor, sample_rate: float):
        """
        Applies batched gain to the input waveform.

        Args:
            audio (torch.Tensor): The audio waveform tensor.
                audio shape: (*batch_dim, num_channels, num_samples)
            sample_rate (float): The sample rate of the audio waveform.

        Returns:
            tuple[torch.Tensor, float]: Transformed audio waveform tensor and its sample rate. Output audio will
                have expanded shape: (*batch_dim, num_gains, num_channels, num_samples)
        """
        ratios = torch.pow(10.0, self.gains / 20.0).to(audio.device).float()
        # Do computations on floats and recast to original type.
        # Carefully handle clipping on integer types
        orig_type = audio.dtype
        audio = audio.float()
        augmented_audio = torch.einsum("...cs,g->...gcs", audio, ratios)

        if orig_type in (torch.int8, torch.int16, torch.int32, torch.int64):
            info = torch.iinfo(orig_type)
            augmented_audio = torch.clamp(augmented_audio, min=info.min, max=info.max)
            augmented_audio = augmented_audio.round().to(orig_type)
        else:
            # augmented_audio = torch.clamp(augmented_audio, min=-1.0, max=1.0).to(
            # orig_type
            # )
            pass

        return augmented_audio, sample_rate
