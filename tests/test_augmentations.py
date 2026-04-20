"""Loads sample audio and ensures each audio transformation method works"""

from itertools import product
from pathlib import Path

import librosa as lr
import numpy as np
import sounddevice
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

    audio = torch.from_numpy(audio[None, :]).float()
    audio_with_sr = (audio, sr)
    return audio_with_sr


def play_audio(audio_with_sr: tuple[torch.Tensor, float]):
    audio = audio_with_sr[0].numpy()
    bsz, n_augs = audio.shape[:2]
    for b, a in product(range(bsz), range(n_augs)):
        print(f"Playing augmentation {a} of batch item {b}")
        sounddevice.play(audio[b, a], samplerate=audio_with_sr[1], blocking=True)


def test_gain_floatingpoint():
    audio_with_sr = make_test_audio()  # (batch, channels, samples)

    gains = np.linspace(-10, 10, 11, endpoint=True)
    transform = audiomanifolds.transformations.Gain(gains=gains)

    orig_shape = audio_with_sr[0].shape

    augmented_audio = transform(audio_with_sr)

    assert augmented_audio[0].shape == (*orig_shape[:-1], len(gains), orig_shape[-1])

    quotients = np.zeros(len(gains) - 1)
    for i in range(len(gains) - 1):
        rms_a = torch.sqrt(torch.mean(augmented_audio[0][..., i, :] ** 2))
        rms_b = torch.sqrt(torch.mean(augmented_audio[0][..., i + 1, :] ** 2))
        quotients[i] = (rms_b / rms_a).item()

    assert np.allclose(quotients, quotients[0])
    if __name__ == "__main__":
        # Do not play audio during pytest
        play_audio(augmented_audio)


def test_gain_integer():
    audio_with_sr = make_test_audio()  # (batch, channels, samples)
    # convert audio to int16
    audio_int16 = (audio_with_sr[0] * 32767).to(torch.int16)
    audio_with_sr = (audio_int16, audio_with_sr[1])

    gains = list(range(-10, 11))
    transform = audiomanifolds.transformations.Gain(gains=gains)

    orig_shape = audio_with_sr[0].shape

    augmented_audio = transform(audio_with_sr)

    assert augmented_audio[0].shape == (*orig_shape[:-1], len(gains), orig_shape[-1])

    quotients = np.zeros(len(gains) - 1)
    float_audio = (
        augmented_audio[0].float() / 32767.0
    )  # convert back to float for RMS calculation
    for i in range(len(gains) - 1):
        rms_a = torch.sqrt((float_audio[..., i, :] ** 2).mean())
        rms_b = torch.sqrt((float_audio[..., i + 1, :] ** 2).mean())
        quotients[i] = (rms_b / rms_a).item()

    # may have some distortion
    assert np.allclose(quotients, quotients[0], atol=0.1)
    if __name__ == "__main__":
        # Do not play audio during pytest
        play_audio(augmented_audio)


def test_time_stretching():
    # Mostly testing that it doesn't crash
    audio_with_sr = make_test_audio()  # (batch, channels, samples)

    ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    transform = audiomanifolds.transformations.TimeStretching(ratios=ratios)

    orig_shape = audio_with_sr[0].shape

    augmented_audio = transform(audio_with_sr)
    assert augmented_audio[0].shape[-1] == orig_shape[-1]  # length preserved
    assert augmented_audio[0].shape == (
        *orig_shape[:-1],
        len(ratios),
        augmented_audio[0].shape[-1],
    )

    if __name__ == "__main__":
        # Do not play audio during pytest
        play_audio(augmented_audio)


def test_pitch_shifting():
    # Mostly testing that it doesn't crash
    audio_with_sr = make_test_audio()  # (batch, channels, samples)

    n_steps = [-12, -9.25, -5, -2, 0, 2, 5, 11.5, 12]
    transform = audiomanifolds.transformations.PitchShifting(n_steps=n_steps)

    orig_shape = audio_with_sr[0].shape

    augmented_audio = transform(audio_with_sr)
    assert augmented_audio[0].shape == (
        *orig_shape[:-1],
        len(n_steps),
        orig_shape[-1],
    )
    if __name__ == "__main__":
        # Do not play audio during pytest
        play_audio(augmented_audio)


def test_lowpass_filter():
    # Mostly testing that it doesn't crash and that shapes are correct.
    audio_with_sr = make_test_audio()

    cutoff_freqs = np.geomspace(100, 15001, 7)
    transform = audiomanifolds.transformations.LowPassFilter(cutoff_freqs)
    orig_shape = audio_with_sr[0].shape

    augmented_audio = transform(audio_with_sr)
    assert augmented_audio[0].shape == (
        *orig_shape[:-1],
        len(cutoff_freqs),
        orig_shape[-1],
    )

    if __name__ == "__main__":
        # Do not play audio during pytest
        play_audio(augmented_audio)


if __name__ == "__main__":
    # Running without pytest to hear augmented audio
    print("Testing Gain with floating point audio...")
    test_gain_floatingpoint()
    print("Testing Gain with integer audio...")
    test_gain_integer()
    print("Testing Time Stretching...")
    test_time_stretching()
    print("Testing Pitch Shifting...")
    test_pitch_shifting()
    print("Testing Low-pass Filter...")
    test_lowpass_filter()
