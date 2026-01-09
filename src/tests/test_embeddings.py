"""Loads sample audio and computes embeddings through each pretrained model"""

from pathlib import Path

import librosa as lr
import numpy as np
import pytest
import soundfile as sf
import torch

from embeddings import interfaces

# from librosa import


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


def test_PANN():
    model = interfaces.PannEmbedder.from_pretrained()
    audio_with_sr = make_test_audio()

    embeddings = model(audio_with_sr)

    assert embeddings.shape == (5, 2048)


def test_Wav2Vec2Base():
    model = interfaces.Wav2Vec2Base.from_pretrained()
    audio_with_sr = make_test_audio()

    embeddings = model(audio_with_sr)

    assert embeddings.shape == (5, model.embedding_dim)
