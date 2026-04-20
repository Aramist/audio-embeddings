import typing as tp

import encodec
import torch
from encodec.utils import convert_audio
from torch import nn
from torch.nn import functional as F

from .base import AudioEmbedder


class EncodecEmbedder(AudioEmbedder):
    def __init__(
        self,
        auto_convert_sample_rate: bool = False,
        sample_rate: tp.Literal["24k", "48k"] = "24k",
        bit_rate: float = 6.0,
    ):
        super().__init__(auto_convert_sample_rate=auto_convert_sample_rate)
        if sample_rate == "24k":
            self.model = encodec.model.EncodecModel.encodec_model_24khz()
        else:
            self.model = encodec.model.EncodecModel.encodec_model_48khz()
        # Allowed values
        # 24k: 1.5 (n_q=2), 3.0 (4), 6.0 (8), 12.0 (16)
        # 48k: 3.0 (n_q=2), 6.0 (n_q=4), 12.0 (n_q=8), 24.0 (n_q=16)
        if sample_rate == "24k" and bit_rate not in [1.5, 3.0, 6.0, 12.0, 24.0]:
            raise ValueError(
                f"Invalid bit rate {bit_rate} for sample rate {sample_rate}. "
                f"Allowed values are [1.5, 3.0, 6.0, 12.0, 24.0]."
            )
        if sample_rate == "48k" and bit_rate not in [3.0, 6.0, 12.0, 24.0]:
            raise ValueError(
                f"Invalid bit rate {bit_rate} for sample rate {sample_rate}. "
                f"Allowed values are [3.0, 6.0, 12.0, 24.0]."
            )

        self.model.set_target_bandwidth(bit_rate)
        self.expected_sample_rate = self.model.sample_rate
        self.embedding_dim = 128

    @classmethod
    def from_pretrained(
        cls,
        auto_convert_sample_rate: bool = False,
        sample_rate: tp.Literal["24k", "48k"] = "24k",
        bitrate: float = 6.0,
    ) -> "EncodecEmbedder":
        """Download/load pretrained weights and return an instance
        of the embedder with those weights loaded.
        """
        model = cls(
            auto_convert_sample_rate=auto_convert_sample_rate,
            sample_rate=sample_rate,
            bit_rate=bitrate,
        )
        model.model.eval()
        return model

    def embed(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute embeddings for a batch of audio samples.

        Args:
            audio (torch.Tensor): Tensor of shape (batch_size, num_samples)

        Returns:
            torch.Tensor: Embeddings of shape (batch_size, embedding_dim)
        """

        # encodec expects a channel dim
        audio = audio.unsqueeze(1)  # (batch_size, 1, num_samples)
        # Returns a list of tuples (encoded_frames, scale_factor)
        # Encoded frames have shape (batch, num_codebooks, num_frames)
        encoded_frames = self.model.encode(audio)
        encoded_frames = [
            self.model.quantizer.decode(frame[0].transpose(0, 1))
            for frame in encoded_frames
        ]
        encoded_frames = torch.cat(encoded_frames, dim=-1)  # (B, C, T)

        # Temporally average to get a fixed-size embedding per audio clip
        return encoded_frames.mean(dim=-1)  # (B, embedding_dim)
