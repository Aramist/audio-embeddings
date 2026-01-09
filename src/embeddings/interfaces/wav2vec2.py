from tempfile import NamedTemporaryFile

import fairseq.checkpoint_utils
import requests
import torch
import tqdm
from torch import nn

from .base import AudioEmbedder

# Links to non-finetuned models
URLS = {
    "base": r"https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt",
    "large": r"https://dl.fbaipublicfiles.com/fairseq/wav2vec/libri960_big.pt",
    "large-lv60": r"https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_new.pt",
}
W2V2_SR = 16000  # Wav2Vec2 standard sampling rate


class __Wav2Vec2Abstract(AudioEmbedder):
    expected_sample_rate: float | None = W2V2_SR
    embedding_dim: int | None = None
    model: nn.Module | None

    def __init__(self):
        super().__init__()
        self.model = None

    @classmethod
    def from_pretrained(cls, url: str | None = None) -> AudioEmbedder:
        if url is None:
            url = URLS["base"]
        with NamedTemporaryFile() as ctx:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get("content-length", 0))
            block_size = 1024
            with tqdm.tqdm(total=total_size, unit="B", unit_scale=True) as prog_bar:
                for data in response.iter_content(block_size):
                    ctx.write(data)
                    prog_bar.update(len(data))
            ctx.flush()
            pseudo_fpath = ctx.name
            embedder = cls()
            test = torch.load(pseudo_fpath, weights_only=False, map_location="cpu")
            # model = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            #     [pseudo_fpath]
            # )[0]
        breakpoint()

    def embed(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute embeddings for the given audio input.

        Args:
            audio (torch.Tensor): Input audio tensor of shape (batch_size, num_channels, num_samples).
        Returns:
            torch.Tensor: Output embeddings of shape (batch_size, embedding_dim).
        """
        if self.model is None:
            raise ValueError(
                "Wav2Vec2 model improperly loaded. Please use from_pretrained()."
            )
        with torch.no_grad():
            z, c = self.model(audio)

        # Not sure which representation should be used downstream
        return c.mean(dim=-1).cpu().numpy()


class Wav2Vec2Base(__Wav2Vec2Abstract):
    embedding_dim = None  # figure out later

    @classmethod
    def from_pretrained(cls) -> AudioEmbedder:
        return super().from_pretrained(URLS["base"])


class Wav2Vec2Large(__Wav2Vec2Abstract):
    embedding_dim = None  # figure out later

    @classmethod
    def from_pretrained(cls) -> AudioEmbedder:
        return super().from_pretrained(URLS["large"])


class Wav2Vec2LargeLV60(__Wav2Vec2Abstract):
    embedding_dim = None  # figure out later

    @classmethod
    def from_pretrained(cls) -> AudioEmbedder:
        return super().from_pretrained(URLS["large-lv60"])
