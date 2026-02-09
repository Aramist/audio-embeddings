"""Loads sample audio and ensures each audio transformation method works"""

from pathlib import Path

import librosa as lr
import numpy as np
import soundfile as sf
import torch

import audiomanifolds.transformations

TEST_AUDIO_PATH = Path(__file__).parent / "steelpan.wav"


def make_test_audio() -> tuple[torch.Tensor, float]:
    """Loads a 5-sec clip of a steelpan performance
    Source: https://www.youtube.com/watch?v=-3Kv4fdm7Uk (AudioSet)

    Returns:
        tuple[torch.Tensor, float]: Audio tensor and its sample rate
    """
    # note: SF reads audio as (samples, channels)
    audio, sr = sf.read(TEST_AUDIO_PATH)  # 5 seconds of music
    audio = audio.mean(axis=1)

    # Generate augmented versions of the audio for comparison
    aug_audio = np.stack(
        [audio]
        + [lr.effects.pitch_shift(audio, sr=sr, n_steps=i) for i in range(1, 5)],
        axis=0,
    )
    aug_audio = aug_audio[:, None, :]  # (batch, channel, samples)
    aug_audio = torch.from_numpy(aug_audio).float()
    audio_with_sr = (aug_audio, sr)
    return audio_with_sr


def test_gain_floatingpoint():
    audio_with_sr = make_test_audio()  # (batch, channels, samples)
    print(audio_with_sr[0].min(), audio_with_sr[0].max())

    gains = np.linspace(-10, 10, 11, endpoint=True)
    transform = audiomanifolds.transformations.Gain(gains=gains)

    orig_shape = audio_with_sr[0].shape

    augmented_audio = transform(audio_with_sr)

    assert augmented_audio[0].shape == (*orig_shape[:-2], len(gains), *orig_shape[-2:])

    quotients = np.zeros(len(gains) - 1)
    for i in range(len(gains) - 1):
        rms_a = torch.sqrt(torch.mean(augmented_audio[0][..., i, :, :] ** 2))
        rms_b = torch.sqrt(torch.mean(augmented_audio[0][..., i + 1, :, :] ** 2))
        quotients[i] = (rms_b / rms_a).item()

    assert np.allclose(quotients, quotients[0])


def test_gain_integer():
    audio_with_sr = make_test_audio()  # (batch, channels, samples)
    # convert audio to int16
    audio_int16 = (audio_with_sr[0] * 32767).to(torch.int16)
    audio_with_sr = (audio_int16, audio_with_sr[1])

    gains = list(range(-10, 11))
    transform = audiomanifolds.transformations.Gain(gains=gains)

    orig_shape = audio_with_sr[0].shape

    augmented_audio = transform(audio_with_sr)

    assert augmented_audio[0].shape == (*orig_shape[:-2], len(gains), *orig_shape[-2:])

    quotients = np.zeros(len(gains) - 1)
    for i in range(len(gains) - 1):
        rms_a = torch.sqrt((augmented_audio[0][..., i, :, :].float() ** 2).mean())
        rms_b = torch.sqrt((augmented_audio[0][..., i + 1, :, :].float() ** 2).mean())
        quotients[i] = (rms_b / rms_a).item()

    # may have some distortion
    assert np.allclose(quotients, quotients[0], atol=0.1)


def test_time_stretching():
    # Mostly testing that it doesn't crash
    audio_with_sr = make_test_audio()  # (batch, channels, samples)

    ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    transform = audiomanifolds.transformations.TimeStretching(ratios=ratios)

    orig_shape = audio_with_sr[0].shape

    augmented_audio = transform(audio_with_sr)
    assert augmented_audio[0].shape[-1] == orig_shape[-1]  # length preserved
    assert augmented_audio[0].shape == (
        *orig_shape[:-2],
        len(ratios),
        *augmented_audio[0].shape[-2:],
    )


def test_pitch_shifting():
    # Mostly testing that it doesn't crash
    audio_with_sr = make_test_audio()  # (batch, channels, samples)

    n_steps = [-12, -9.25, -5, -2, 0, 2, 5, 11.5, 12]
    transform = audiomanifolds.transformations.PitchShifting(n_steps=n_steps)

    orig_shape = audio_with_sr[0].shape

    augmented_audio = transform(audio_with_sr)
    assert augmented_audio[0].shape == (
        *orig_shape[:-2],
        len(n_steps),
        *orig_shape[-2:],
    )
