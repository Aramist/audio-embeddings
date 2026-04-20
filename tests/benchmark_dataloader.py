import argparse
import os
import time
import typing as tp
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from audiomanifolds import transformations

args_for_augs = {
    "gain": {"gains": np.linspace(-10, 10, 101, endpoint=True)},
    "time_stretching": {
        "ratios": np.exp(
            np.linspace(
                np.log(0.5),
                np.log(2.0),
                101,
            )
        )
    },
    "pitch_shifting": {"n_steps": np.linspace(-12, 12, 101, endpoint=True)},
    "low_pass_filter": {
        "cutoff_frequencies": np.ascontiguousarray(
            np.geomspace(100, 15000, 100, endpoint=True)[::-1]
        )  # Reversed so the less-impactful augmentations come first
    },
}

module_lookup = {
    "gain": transformations.Gain,
    "time_stretching": transformations.TimeStretching,
    "pitch_shifting": transformations.PitchShifting,
    "low_pass_filter": transformations.LowPassFilter,
}


class AudioDataset(Dataset):

    def __init__(
        self,
        audio_paths: list[Path],
        target_sr: float,
        clip_length: float | None = None,
        augmentation: tp.Callable | None = None,
    ):
        self.audio_paths = audio_paths
        self.target_sr = target_sr
        self.clip_length = clip_length
        self.augmentation = augmentation

    def __len__(self):
        return len(self.audio_paths)

    def _load_audio(self, audio_path: Path) -> tuple[torch.Tensor, float]:
        audio, sr = torchaudio.load(audio_path)
        audio = audio[:1, :]  # (1 channel, samples)

        if self.clip_length is not None:
            clip_length_samples = int(sr * self.clip_length)
            audio = audio[:, :clip_length_samples]

        return (audio, sr)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        """Grabs audio sample and its sample rate

        Args:
            idx (int): Index of the audio sample to grab (within the audio_paths list)

        Returns:
            tuple[torch.Tensor, float]: Audio and sample rate
        """
        audio_path = self.audio_paths[idx]
        audio_tensor, sr = self._load_audio(audio_path)
        if self.augmentation is not None:
            audio_tensor, sr = self.augmentation((audio_tensor, sr))

        return (audio_tensor, sr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "augmentation",
        type=str,
        help="Type of augmentation to apply",
        choices=list(args_for_augs.keys()),
    )
    augmentation = ap.parse_args().augmentation
    BSD_AUDIO_PATH = Path("/Users/aramis/audio/")

    print(f"Using CUDA: {torch.cuda.is_available()}")
    audio_paths = sorted(BSD_AUDIO_PATH.glob("*.wav"))
    kwargs = args_for_augs[augmentation]

    augment_module: transformations.AudioTransformation

    if augmentation not in module_lookup:
        raise ValueError(f"Unknown augmentation: {augmentation}")

    augment_module_class = module_lookup[augmentation]
    augment_module = augment_module_class(**kwargs)

    sr = 16000.0

    dset = AudioDataset(audio_paths, target_sr=sr, augmentation=None, clip_length=10.0)
    try:
        num_avail_cpu = len(os.sched_getaffinity(0))
    except AttributeError:
        num_avail_cpu = os.cpu_count() or 1
    num_workers = max(1, num_avail_cpu - 2)

    dloader = DataLoader(
        dset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    num_samples = 50
    start_time = time.time()
    for data, _ in tqdm(zip(dloader, range(num_samples)), total=num_samples):
        audio, sr = data
        # Data: (batch, channels, samples)
        audio = audio.squeeze(1)
        sr = sr.item()
        aug_data = augment_module((audio, sr))
    end_time = time.time()
    total_time = end_time - start_time
    print(
        f"Time per sample ({num_samples} samples): {total_time / num_samples:.2f} seconds"
    )
