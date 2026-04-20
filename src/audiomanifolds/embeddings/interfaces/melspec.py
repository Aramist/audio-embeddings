import typing as tp

import torch
import torchaudio

from .base import AudioEmbedder


class MelSpecEmbedder(AudioEmbedder):
    def __init__(
        self,
        auto_convert_sample_rate: bool = False,
        n_fft: int = 2048,
        n_mels: int = 128,
    ):
        super().__init__(auto_convert_sample_rate=auto_convert_sample_rate)
        self.n_fft = n_fft
        self.embedding_dim = n_mels

    @classmethod
    def from_pretrained(
        cls,
        auto_convert_sample_rate: bool = False,
    ) -> "MelSpecEmbedder":
        """Download/load pretrained weights and return an instance
        of the embedder with those weights loaded.
        """
        model = cls(
            auto_convert_sample_rate=auto_convert_sample_rate,
        )
        return model

    def embed(self, audio: torch.Tensor, sample_rate: float) -> torch.Tensor:
        """Compute mel spectrogram embeddings for the given audio input.

        Args:
            audio (torch.Tensor): 2D input audio tensor of shape (batch_size, num_samples).
            sample_rate (float): The sample rate of the input audio.
        Returns:
            torch.Tensor: Output embeddings of shape (batch_size, embedding_dim).
        """

        # Settings mostly taken from LAION-CLAP
        # https://github.com/LAION-AI/CLAP/blob/1fd4c37df5ffbfcfbad5415c170bc66cf94c9994/src/laion_clap/training/data.py#L363
        mel_spec_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=int(sample_rate),
            n_fft=self.n_fft,
            hop_length=self.n_fft // 4,
            n_mels=self.embedding_dim,
            power=2.0,
            norm=None,
            pad_mode="reflect",
        )
        mel_spec = mel_spec_transform(audio)
        # Averaged over time
        return torchaudio.transforms.AmplitudeToDB()(mel_spec).mean(
            dim=-1
        )  # (*batch, n_mels)

    def forward(self, audio: tuple[torch.Tensor, float]) -> torch.Tensor:
        """Forward pass through the embedder. Overridden to allow any sample rate

        Args:
            audio (tuple[torch.Tensor, float]): A tuple containing the input tensor of shape (*batch_dims, num_samples)
            and the sample rate as a float.
        Returns:
            torch.Tensor: Output embeddings of shape (*batch_dims, embedding_dim).
        """

        if not isinstance(audio, tuple) or len(audio) != 2:
            raise ValueError("Input audio must be a tuple of (tensor, sample_rate).")

        signal = audio[0]
        batch_dims = ()
        if signal.ndim == 1:
            # Missing batch dimensions
            signal = signal.unsqueeze(0)
        elif signal.ndim > 2:
            batch_dims = signal.shape[:-1]
            signal = signal.reshape(-1, signal.shape[-1])

        signal = signal.float()
        with torch.no_grad():
            embedding = self.embed(signal, sample_rate=audio[1])
        embedding = embedding.reshape(*batch_dims, embedding.shape[-1])

        return embedding
