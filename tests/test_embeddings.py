"""Loads sample audio and computes embeddings through each pretrained model"""

from pathlib import Path

import librosa as lr
import numpy as np
import soundfile as sf
import torch

from audiomanifolds.embeddings import CLAPAudioEmbedder, PannEmbedder

TEST_AUDIO_PATH = Path(__file__).parent / "steelpan.wav"


def make_test_audio(
    target_sample_rate: float | None = None,
) -> tuple[torch.Tensor, float]:
    """Loads a 5-sec clip of a steelpan performance
    Source: https://www.youtube.com/watch?v=-3Kv4fdm7Uk (AudioSet)

    Returns:
        tuple[torch.Tensor, float]: Audio tensor and its sample rate
    """
    # note: SF reads audio as (samples, channels)
    audio, sr = sf.read(TEST_AUDIO_PATH)  # 5 seconds of music
    audio = audio.mean(axis=1)

    if target_sample_rate is not None and abs(sr - target_sample_rate) > 1e-3:
        audio = lr.resample(audio, orig_sr=sr, target_sr=target_sample_rate)
        sr = target_sample_rate

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
    model = PannEmbedder.from_pretrained()
    audio_with_sr = make_test_audio(model.expected_sample_rate)

    embeddings = model(audio_with_sr)

    assert embeddings.shape == (*audio_with_sr[0].shape[:-1], 2048)


def test_CLAP():
    model = CLAPAudioEmbedder.from_pretrained()
    audio_with_sr = make_test_audio(model.expected_sample_rate)

    embeddings = model(audio_with_sr)

    assert embeddings.cpu().numpy().shape == (*audio_with_sr[0].shape[:-1], 512)
