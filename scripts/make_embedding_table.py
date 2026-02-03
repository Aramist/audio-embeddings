import argparse
import time
import typing as tp
from pathlib import Path

import h5py
import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
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


def run(
    audio_dir: Path,
    augmentation: str,
    clip_length: float | None = None,
    save_to: Path | None = None,
):
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
    if save_to is None:
        save_to = Path("embeddings.h5")
    with h5py.File(save_to, "w") as hf:
        hf.create_dataset("embeddings", data=embeddings.numpy())


def visualize_embeddings(embedding_file: Path):
    with h5py.File(embedding_file, "r") as hf:
        embeddings: np.ndarray = hf["embeddings"][:]
    print("Loaded embeddings shape: ", embeddings.shape)
    num_samples, num_augs, embedding_dim = embeddings.shape
    # Make each augmentation relative to the original audio
    orig_audio_index = num_augs // 2
    orig_audio_embeddings = embeddings[:, orig_audio_index, :]  # (50, embedding_dim)
    centroids = embeddings.mean(axis=1)

    # Compute covariance matrices within each class (audio file)
    # Target shape: (num_classes, num_features, num_features)
    mean_centered = (
        embeddings - centroids[:, None, :]
    )  # (num_classes, num_augs, features)
    cov = np.einsum("cai,cak->cik", mean_centered, mean_centered) / num_augs

    within_class_cov = cov.mean(axis=0)
    # shape: (features, features)

    # shape: (num_classes, features)
    mean_centered_centroids = orig_audio_embeddings - orig_audio_embeddings.mean(
        axis=0, keepdims=True
    )
    # shape: (features, features)
    across_class_cov = mean_centered_centroids.T @ mean_centered_centroids

    reg = np.eye(2048) * 1e-4
    generalized_eig, generalized_eigv = scipy.linalg.eigh(
        a=within_class_cov + reg, b=across_class_cov + reg, driver="gv"
    )
    # generalized_eig, generalized_eigv = scipy.linalg.eigh(a=within_class_cov)

    # Reverse them so largest come first
    generalized_eigv = (generalized_eigv.T)[::-1]  # shape (num_eigv, dim)
    generalized_eig = generalized_eig[::-1]

    top_two_directions = generalized_eigv[:2]  # shape: (2, features)

    embeddings = embeddings - embeddings[:, orig_audio_index, :][:, None, :]
    projected_embeddings = np.einsum(
        "df,cbf->cbd", top_two_directions, embeddings
    )  # (num_files, num_augs, features)

    cumulative_eigenvalues = np.cumsum(generalized_eig)
    cutoff_idx = np.flatnonzero(
        (cumulative_eigenvalues / cumulative_eigenvalues[-1]) > 0.95
    )[0]
    print(cutoff_idx)
    generalized_eig = generalized_eig[:cutoff_idx]
    generalized_eigv = generalized_eigv[:cutoff_idx]
    print(generalized_eig.shape, generalized_eigv.shape)

    # fig, ax = plt.subplots()
    # ax.plot(generalized_eig)
    # ax.set_xlabel("Eigenvalue index")
    # ax.set_ylabel("Eigenvalue")
    # ax.set_yscale("log")
    # plt.show()
    # exit()
    embeddings = embeddings.reshape(num_samples * num_augs, embedding_dim)
    projected_embeddings = projected_embeddings.reshape(num_samples * num_augs, 2)

    # scaler = StandardScaler()
    # embeddings = scaler.fit_transform(embeddings)

    pca = PCA(n_components=2)
    # embeddings_2d = pca.fit_transform(embeddings)
    embeddings_2d = projected_embeddings

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
    aug_type = "_".join(embedding_file.stem.split("_")[2:])
    cbar.set_label(f"{aug_type} strength")
    cbar.set_ticks([0, 0.5, 1])
    if aug_type == "gain":
        low_gain = f'{args_for_augs["gain"]["gains"][0]:.0f} dB'
        mid_gain = "0 dB"
        hi_gain = f'{args_for_augs["gain"]["gains"][-1]:.0f} dB'
        cbar.set_ticklabels([low_gain, mid_gain, hi_gain])
    elif aug_type == "time_stretching":
        low_ratio = f'{args_for_augs["time_stretching"]["ratios"][0]:.2f}x'
        mid_ratio = "1.00x"
        hi_ratio = f'{args_for_augs["time_stretching"]["ratios"][-1]:.2f}x'
        cbar.set_ticklabels([low_ratio, mid_ratio, hi_ratio])
    elif aug_type == "pitch_shifting":
        low_steps = f'{args_for_augs["pitch_shifting"]["n_steps"][0]:.0f} st'
        mid_steps = "0 st"
        hi_steps = f'{args_for_augs["pitch_shifting"]["n_steps"][-1]:.0f} st'
        cbar.set_ticklabels([low_steps, mid_steps, hi_steps])
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("audio_dir", type=Path)
    ap.add_argument("augmentation", type=str, choices=list(args_for_augs.keys()))
    ap.add_argument("--clip-length", type=float, default=1.0)
    ap.add_argument("-o", "--output-dir", type=Path, default=Path("."))
    args = ap.parse_args()

    dataset_name = args.audio_dir.stem
    augmentation_name = args.augmentation
    output_path = (
        args.output_dir / f"embedding_table_{dataset_name}_{augmentation_name}.h5"
    )
    if not Path(output_path).exists():
        args.output_dir.mkdir(parents=True, exist_ok=True)
        run(
            args.audio_dir,
            args.augmentation,
            clip_length=args.clip_length,
            save_to=output_path,
        )

    visualize_embeddings(output_path)
