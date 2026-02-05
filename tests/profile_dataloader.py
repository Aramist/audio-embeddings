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
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from audiomanifolds import embeddings, transformations

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
}

module_lookup = {
    "gain": transformations.Gain,
    "time_stretching": transformations.TimeStretching,
    "pitch_shifting": transformations.PitchShifting,
}


class AudioDataset(Dataset):
    cached_audio: list[tuple[torch.Tensor, float]] | None = None

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

        if AudioDataset.cached_audio is None:
            AudioDataset.cached_audio = [
                self._load_audio(p)
                for p in tqdm(audio_paths[:1000], desc="Caching audio...")
            ]

    def __len__(self):
        return len(self.audio_paths)

    def _load_audio(self, audio_path: Path) -> tuple[torch.Tensor, float]:
        audio, sr = sf.read(audio_path, always_2d=True)
        audio = audio[:, 0]  # use only one channel
        audio = audio[None, :]  # (channels, samples)
        if abs(sr - self.target_sr) > 1e-3:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr

        if self.clip_length is not None:
            clip_length_samples = int(sr * self.clip_length)
            audio = audio[..., :clip_length_samples]

        if audio.dtype in (np.int16, np.int32):
            max_val = np.iinfo(audio.dtype).max
            audio = audio.astype(np.float32) / max_val

        audio_tensor = torch.from_numpy(audio).float()
        return (audio_tensor, sr)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float]:
        """Grabs audio sample and its sample rate

        Args:
            idx (int): Index of the audio sample to grab (within the audio_paths list)

        Returns:
            tuple[torch.Tensor, float]: Audio and sample rate
        """
        audio_path = self.audio_paths[idx]
        # assume cached_audio exists
        if AudioDataset.cached_audio is None:
            audio_tensor, sr = self._load_audio(audio_path)
            # raise RuntimeError(
            #     "AudioDataset.cached_audio is None, but it should be populated."
            # )
        else:
            audio_tensor, sr = AudioDataset.cached_audio[idx]
        if self.augmentation is not None:
            audio_tensor, sr = self.augmentation((audio_tensor, sr))

        return (audio_tensor, sr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "augmentation",
        type=str,
        help="Type of augmentation to apply",
        choices=["gain", "time_stretching", "pitch_shifting"],
    )
    augmentation = ap.parse_args().augmentation
    BSD_AUDIO_PATH = Path("/ext3/BSD10k_audio/")
    BSD_METADATA_PATH = Path("/ext3/bsd_id_to_class_mapping.csv")
    output_dir = Path("/scratch/at4219/")

    print(f"Using CUDA: {torch.cuda.is_available()}")
    # get lengths of all audio files to filter for longer clips
    audio_paths = sorted(BSD_AUDIO_PATH.glob("*.wav"))
    class_id_df = pd.read_csv(BSD_METADATA_PATH)
    if "durations" not in class_id_df.columns:
        durations = {}
        for audio_path in tqdm(audio_paths, desc="Getting audio durations..."):
            info = sf.info(audio_path)
            sound_id = int(audio_path.stem)
            durations[sound_id] = info.frames / info.samplerate
        class_id_df["durations"] = class_id_df["sound_id"].map(durations)
        class_id_df.to_csv(BSD_METADATA_PATH, index=False)
    else:
        durations = {
            int(row["sound_id"]): row["durations"] for _, row in class_id_df.iterrows()
        }
    # filtered_audio_paths = list(
    #     filter(lambda p: durations[int(p.stem)] >= 1.0, audio_paths)
    # )
    filtered_audio_paths = audio_paths

    kwargs = args_for_augs[augmentation]

    augment_module: transformations.AudioTransformation

    if augmentation not in module_lookup:
        raise ValueError(f"Unknown augmentation: {augmentation}")

    augment_module_class = module_lookup[augmentation]
    augment_module = augment_module_class(**kwargs)

    sr = 32000.0

    dset = AudioDataset(
        audio_paths, target_sr=sr, augmentation=augment_module, clip_length=10.0
    )
    try:
        num_avail_cpu = len(os.sched_getaffinity(0))
    except AttributeError:
        num_avail_cpu = os.cpu_count() or 1
    num_workers = max(1, num_avail_cpu - 2)
    dloader = DataLoader(
        dset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )

    start_time = time.time()
    for data, _ in tqdm(zip(dloader, range(1000)), total=1000):
        pass
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time for 1000 samples: {total_time:.2f} seconds")
