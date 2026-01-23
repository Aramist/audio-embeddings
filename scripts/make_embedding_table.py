import argparse
import time
import typing as tp
from pathlib import Path

import h5py
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import audiomanifolds.embeddings
import audiomanifolds.transformations

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
    "gain": audiomanifolds.transformations.Gain,
    "time_stretching": audiomanifolds.transformations.TimeStretching,
    "pitch_shifting": audiomanifolds.transformations.PitchShifting,
}


def load_audio(
    audio_path: Path, crop_length: float | None = None
) -> tuple[torch.Tensor, float]:
    """Loads an audio file and returns its waveform and sample rate

    Returns:
        tuple[torch.Tensor, float]: Audio tensor (batch, samples) and its sample rate
    """
    # note: SF reads audio as (samples, channels)
    audio, sr = sf.read(audio_path, always_2d=True)  # 5 seconds of music
    # audio = audio.mean(axis=1)
    audio = audio[:, 0]  # only use one channel

    if crop_length is not None:
        audio = audio[: min(len(audio), int(sr * crop_length))]

    audio = audio[None, None, :]  # (batch, channels, samples)
    audio = torch.from_numpy(audio).float()
    audio_with_sr = (audio, sr)
    return audio_with_sr


def convert_sr(
    audio: tuple[torch.Tensor, float], target_sr: float
) -> tuple[torch.Tensor, float]:
    """Converts audio to target sample rate

    Returns:
        tuple[torch.Tensor, float]: Audio tensor (batch, samples) and its sample rate
    """

    if abs(target_sr - audio[1]) < 1e-3:
        return audio

    orig_device = audio[0].device
    orig_dtype = audio[0].dtype
    waveform = audio[0].cpu().numpy()

    new_waveform = librosa.resample(waveform, orig_sr=audio[1], target_sr=target_sr)
    new_waveform = torch.from_numpy(new_waveform).to(orig_device).to(orig_dtype)

    return new_waveform, target_sr


def run(audio_dir: Path, augmentation: str, clip_length: float | None = None):
    kwargs = args_for_augs[augmentation]

    loaded_audio = [
        load_audio(audio_path) for audio_path in sorted(audio_dir.glob("*.wav"))
    ]

    if not all([sr == loaded_audio[0][1] for _, sr in loaded_audio]):
        target_sr = min([sr for _, sr in loaded_audio])
        loaded_audio = [
            convert_sr(audio, target_sr=target_sr) for audio in loaded_audio
        ]

    sr = loaded_audio[0][1]

    clip_len = min([audio.shape[-1] for audio, _ in loaded_audio])
    if clip_length is not None:
        clip_len = min(clip_len, int(sr * clip_length))
    stacked_audio = (
        torch.concatenate([audio[..., :clip_len] for audio, _ in loaded_audio], dim=0),
        sr,
    )  # new shape: (batch, channels, clip_len)

    augment_module: audiomanifolds.transformations.AudioTransformation

    if augmentation not in module_lookup:
        raise ValueError(f"Unknown augmentation: {augmentation}")

    augment_module_class = module_lookup[augmentation]
    augment_module = augment_module_class(**kwargs)

    augmented_audio = augment_module(stacked_audio)
    # augmented_audio shape: (batch, num_augs, channels, clip_len)
    print("Augmented audio: ", augmented_audio[0].shape)

    embedding_module = audiomanifolds.embeddings.PannEmbedder.from_pretrained()
    embedding_module.eval()
    start_time = time.time()
    embeddings = []
    for audio in tqdm(augmented_audio[0], desc="Computing embeddings"):
        embeddings.append(embedding_module((audio, sr)))
    embeddings = torch.stack(embeddings, dim=0)
    end_time = time.time()
    print(f"Embedding computation time: {end_time - start_time:.2f} seconds")
    # embeddings shape: (batch, num_augs, embedding_dim)
    print("Embeddings shape: ", embeddings.shape)
    with h5py.File(f"embedding_table_{augmentation}.h5", "w") as hf:
        hf.create_dataset("embeddings", data=embeddings.numpy())


def visualize_embeddings(embedding_file: Path):
    with h5py.File(embedding_file, "r") as hf:
        embeddings: np.ndarray = hf["embeddings"][:]
    print("Loaded embeddings shape: ", embeddings.shape)
    num_samples, num_augs, embedding_dim = embeddings.shape
    embeddings = embeddings.reshape(num_samples * num_augs, embedding_dim)

    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    colors = np.tile(np.linspace(0, 1, num_augs), num_samples)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=10, c=colors, cmap="RdBu"
    )
    ax.set_title("PCA of Embeddings")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.grid(True)
    ax.axis("equal")

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap="RdBu"), ax=ax, orientation="vertical"
    )
    aug_type = embedding_file.stem.split("_")[-1]
    cbar.set_label(f"{aug_type} strength")
    cbar.set_ticks([0, 0.5, 1])
    if aug_type == "gain":
        low_gain = f'{args_for_augs["gain"]["gains"][0]:.0f} dB'
        mid_gain = "0 dB"
        hi_gain = f'{args_for_augs["gain"]["gains"][-1]:.0f} dB'
        cbar.set_ticklabels([low_gain, mid_gain, hi_gain])
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_dir", type=Path)
    ap.add_argument("augmentation", type=str, choices=list(args_for_augs.keys()))
    ap.add_argument("--clip-length", type=float, default=1.0)
    args = ap.parse_args()

    if not Path(f"embedding_table_{args.augmentation}.h5").exists():
        run(args.audio_dir, args.augmentation, clip_length=args.clip_length)

    visualize_embeddings(Path(f"embedding_table_{args.augmentation}.h5"))
