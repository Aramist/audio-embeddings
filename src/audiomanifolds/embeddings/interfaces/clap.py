"""Based on code from LAION-CLAP repo:
https://github.com/LAION-AI/CLAP
"""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import requests
import torch
from laion_clap.hook import CLAP_Module  # for typing
from laion_clap.training.data import get_audio_features
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from .base import AudioEmbedder


def _load_state_dict(checkpoint_path: Path, map_location="cpu", skip_params=True):
    # https://github.com/LAION-AI/CLAP/blob/1fd4c37df5ffbfcfbad5415c170bc66cf94c9994/src/laion_clap/clap_module/factory.py#L53
    checkpoint = torch.load(
        checkpoint_path, map_location=map_location, weights_only=False
    )
    if (
        isinstance(checkpoint, dict)
        or isinstance(checkpoint, OrderedDict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if skip_params:
        if next(iter(state_dict.items()))[0].startswith("module"):
            # Remove 'module.' prefix from all keys
            state_dict = {k[7:]: v for k, v in state_dict.items()}

    # For some reason, this parameter isn't expected
    del state_dict["text_branch.embeddings.position_ids"]
    return state_dict


class CLAPAudioEmbedder(AudioEmbedder):

    def __init__(
        self, quantize_input: bool = True, auto_convert_sample_rate: bool = False
    ):
        super().__init__(auto_convert_sample_rate=auto_convert_sample_rate)

        self.expected_sample_rate = 48000
        self.embedding_dim = 512
        self.quantize_input = quantize_input

        self.model: CLAP_Module = CLAP_Module(enable_fusion=False)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Load the pretrained CLAP model."""
        instance = cls(*args, **kwargs)
        # Manually re-implementing this because the original hasn't been updated since the
        # weights_only change to PyTorch
        checkpoint_cache_dir = Path(torch.hub.get_dir())
        # Default checkpoint is appropriate for general audio <10 seconds long
        weight_file_name = "630k-audioset-best.pt"
        download_link = (
            f"https://huggingface.co/lukewys/laion_clap/resolve/main/{weight_file_name}"
        )
        checkpoint_path = checkpoint_cache_dir / weight_file_name
        if not checkpoint_path.exists():
            print(
                f"Downloading CLAP weights from {download_link} to {checkpoint_path}..."
            )
            response = requests.get(download_link, stream=True)
            response.raise_for_status()
            bar = tqdm(
                total=int(response.headers.get("Content-Length", 0)),
                unit="B",
                unit_scale=True,
            )
            with open(checkpoint_path, "wb") as checkpoint_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        bar.update(len(chunk))
                        checkpoint_file.write(chunk)
        ckpt = _load_state_dict(checkpoint_path, skip_params=True)
        instance.model.model.load_state_dict(ckpt)
        instance.model.eval()
        instance.eval()

        return instance

    def quantize(self, audio: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        int16_max = float(torch.iinfo(torch.int16).max)
        if isinstance(audio, torch.Tensor):
            return (torch.clamp(audio, -1.0, 1.0) * int16_max).to(
                torch.int16
            ) / int16_max
        else:
            return (np.clip(audio, -1.0, 1.0) * int16_max).astype(np.int16) / int16_max

    def embed(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute embeddings for the given audio input.

        Args:
            audio (torch.Tensor): Input audio tensor of shape (batch_size, num_channels, num_samples).
        Returns:
            torch.Tensor: Output embeddings of shape (batch_size, embedding_dim).
        """
        batch_shape = audio.shape[:-1]
        audio = audio.view(-1, audio.shape[-1])
        if self.quantize_input:
            int16_max = float(torch.iinfo(torch.int16).max)
            audio = (
                (torch.clamp(audio, -1.0, 1.0) * int16_max).to(torch.int16) / int16_max
            ).float()
        audio_input = []
        for waveform in audio:
            temp_dict = {}
            temp_dict = get_audio_features(
                temp_dict,
                waveform,
                480000,
                data_truncating="fusion" if self.model.enable_fusion else "rand_trunc",
                data_filling="repeatpad",
                audio_cfg=self.model.model_cfg["audio_cfg"],
            )
            audio_input.append(temp_dict)
        embeddings = self.model.model.get_audio_embedding(audio_input)
        embeddings = embeddings.view(*batch_shape, self.embedding_dim)
        return embeddings
