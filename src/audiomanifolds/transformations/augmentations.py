import numpy as np
import pedalboard
import torch
from joblib import Parallel, delayed

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
                audio shape: (batch_size, num_samples)
            sample_rate (float): The sample rate of the audio waveform.

        Returns:
            tuple[torch.Tensor, float]: Transformed audio waveform tensor and its sample rate. Output audio will
                have expanded shape: (batch_size, num_gains, num_samples)
        """
        # Add nan as an explicit no-op value
        self.gains[torch.isnan(self.gains)] = 0.0
        ratios = torch.pow(10.0, self.gains / 20.0).to(audio.device).float()
        # Do computations on floats and recast to original type.
        # Carefully handle clipping on integer types
        orig_type = audio.dtype
        audio = audio.float()
        augmented_audio = torch.einsum("bs,g->bgs", audio, ratios)

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


class TimeStretching(AudioTransformation):
    """Applies time stretching or compression to the input audio signal."""

    def __init__(self, ratios: list[float] | np.ndarray | torch.Tensor):
        """
        Args:
            ratios (list[float] | np.ndarray | torch.Tensor): Ratios by which to stretch (ratio > 1) or compress (ratio < 1) the audio signal.
        """

        super(TimeStretching, self).__init__()
        if not isinstance(ratios, torch.Tensor):
            ratios = torch.tensor(ratios)
        self.ratios = ratios

    def apply(self, audio: torch.Tensor, sample_rate: float):
        """
        Applies time stretching/compression to the input waveform.

        Args:
            audio (torch.Tensor): The audio waveform tensor.
                audio shape: (batch_size, num_samples)
            sample_rate (float): The sample rate of the audio waveform.
        Returns:
            tuple[torch.Tensor, float]: Transformed audio waveform tensor and its sample rate. Output audio will
                have expanded shape: (batch_size, num_ratios, num_samples)
        """

        def is_noop(ratio: float) -> bool:
            return np.isnan(ratio) or np.abs(ratio - 1.0) < 1e-3

        audio_np = audio.cpu().numpy()
        _, orig_length = audio_np.shape
        stretched_audios = []
        if len(self.ratios) < 10:
            for ratio in self.ratios:
                if is_noop(ratio.item()):
                    stretched_audios.append(audio_np)
                    continue
                stretched_audios.append(
                    pedalboard.time_stretch(
                        audio_np,
                        samplerate=sample_rate,
                        stretch_factor=ratio.item(),
                        high_quality=False,
                    )
                )
        else:
            stretched_audios = Parallel(n_jobs=-1, return_as="list")(
                [
                    delayed(
                        lambda ratio: (
                            pedalboard.time_stretch(
                                audio_np,
                                samplerate=sample_rate,
                                stretch_factor=ratio.item(),
                                high_quality=False,
                            )
                            if not is_noop(ratio.item())
                            else audio_np
                        )
                    )(ratio)
                    for ratio in self.ratios
                ]
            )

        def pad_or_truncate(stretched_audio):
            # Pad or truncate to original audio length
            new_len = stretched_audio.shape[-1]
            if new_len < orig_length:
                pad_size = orig_length - new_len
                ndim = stretched_audio.ndim
                padding = [(0, 0)] * (ndim - 1) + [(0, pad_size)]
                stretched_audio = np.pad(stretched_audio, padding, "constant")
            elif new_len > orig_length:
                stretched_audio = stretched_audio[..., :orig_length]
            return stretched_audio

        stretched_audios = list(map(pad_or_truncate, stretched_audios))
        stretched_audios = np.stack(
            stretched_audios, axis=1
        )  # (batch, ratios, samples)
        stretched_audios = torch.from_numpy(stretched_audios).to(audio.device)
        return stretched_audios, sample_rate


class PitchShifting(AudioTransformation):
    """Applies pitch shifting to the input audio signal without affecting its duration."""

    def __init__(self, n_steps: list[float] | np.ndarray | torch.Tensor):
        """
        Args:
            n_steps (list[float] | np.ndarray | torch.Tensor): Number of steps to shift the pitch (in semitones).
                Positive values increase pitch, negative values decrease pitch.
        """

        super(PitchShifting, self).__init__()
        if not isinstance(n_steps, torch.Tensor):
            n_steps = torch.tensor(n_steps)
        self.n_steps = n_steps

    def apply(self, audio: torch.Tensor, sample_rate: float):
        """
        Applies pitch shifting to the input waveform.

        Args:
            audio (torch.Tensor): The audio waveform tensor.
                audio shape: (batch_size, num_samples)
            sample_rate (float): The sample rate of the audio waveform.
        Returns:
            tuple[torch.Tensor, float]: Transformed audio waveform tensor and its sample rate. Output audio will
                have expanded shape: (batch_size, num_steps, num_samples)
        """

        def is_noop(n_step: float) -> bool:
            return np.isnan(n_step) or np.abs(n_step) < 1e-3

        audio_np = audio.cpu().numpy()
        if len(self.n_steps) < 10:
            shifted_audios = [
                (
                    pedalboard.PitchShift(n_step.item())(audio_np, sample_rate)
                    if not is_noop(n_step.item())
                    else audio_np
                )
                for n_step in self.n_steps
            ]
            shifted_audios = torch.from_numpy(np.stack(shifted_audios, axis=1)).to(
                audio.device
            )
        else:
            shifted_audios = Parallel(n_jobs=-1, return_as="list")(
                [
                    delayed(
                        lambda n_steps: (
                            torch.from_numpy(
                                pedalboard.PitchShift(semitones=n_steps.item())(
                                    audio_np, sample_rate
                                )
                            )
                            if not is_noop(n_steps.item())
                            else audio_np
                        )
                    )(n_step)
                    for n_step in self.n_steps
                ]
            )

            shifted_audios = torch.stack(shifted_audios, dim=1).to(audio.device)
        return shifted_audios, sample_rate


class LowPassFilter(AudioTransformation):
    """Applies low pass filtering to the input signal"""

    cutoff_frequencies: torch.Tensor

    def __init__(self, cutoff_frequencies: list[float] | np.ndarray | torch.Tensor):
        """
        Args:
            cutoff_frequencies (list[float] | np.ndarray | torch.Tensor): Cutoff frequencies for the low pass filter. (point at which attenuation is -3dB)
        """

        super(LowPassFilter, self).__init__()
        if not isinstance(cutoff_frequencies, torch.Tensor):
            cutoff_frequencies = torch.tensor(cutoff_frequencies)
        self.cutoff_frequencies = cutoff_frequencies

    def apply(self, audio: torch.Tensor, sample_rate: float):
        """
        Applies low pass filters to copies of the input waveform.

        Args:
            audio (torch.Tensor): The audio waveform tensor.
                audio shape: (batch_size, num_samples)
            sample_rate (float): The sample rate of the audio waveform.
        Returns:
            tuple[torch.Tensor, float]: Transformed audio waveform tensor and its sample rate. Output audio will
                have expanded shape: (batch_size, num_filters, num_samples)
        """

        def is_noop(cutoff_freq: float) -> bool:
            return np.isnan(cutoff_freq) or cutoff_freq >= sample_rate / 2.0

        audio_np = audio.detach().cpu().numpy()
        if len(self.cutoff_frequencies) < 10:
            filtered_audios = np.stack(
                [
                    (
                        pedalboard.LowpassFilter(
                            cutoff_frequency_hz=cutoff_freq.item()
                        )(audio_np, sample_rate)
                        if not is_noop(cutoff_freq.item())
                        else audio_np
                    )
                    for cutoff_freq in self.cutoff_frequencies
                ],
                axis=1,
            )
        else:
            jobs = [
                delayed(
                    lambda cutoff_freq: (
                        pedalboard.LowpassFilter(
                            cutoff_frequency_hz=cutoff_freq.item()
                        )(audio_np, sample_rate)
                        if not is_noop(cutoff_freq.item())
                        else audio_np
                    )
                )(cutoff_freq)
                for cutoff_freq in self.cutoff_frequencies
            ]
            filtered_audios = Parallel(n_jobs=-1, return_as="list")(jobs)
            filtered_audios = np.stack(filtered_audios, axis=1)

        return torch.from_numpy(filtered_audios).to(audio.device), sample_rate
