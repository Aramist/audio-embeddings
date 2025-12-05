"""Contains code copied from PANNs (2019)
Original repo: https://github.com/qiuqiangkong/panns_inference
Paper: https://arxiv.org/abs/1912.10211

Modifications:
    - remove rectification from embedding returned by CNN14
    - remove classification head
    - autoformat with Black
"""

import torch
from torch import nn
from torch.nn import functional as F
from torchlibrosa.augmentation import SpecAugmentation
from torchlibrosa.stft import LogmelFilterBank, Spectrogram

from .base import AudioEmbedder


def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type="avg"):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise ValueError("Incorrect argument!")
        return x


class Cnn14(nn.Module):
    def __init__(
        self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num
    ):

        super(Cnn14, self).__init__()

        window = "hann"
        center = True
        pad_mode = "reflect"
        ref = 1.0
        amin = 1e-10
        top_db = None

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=sample_rate,
            n_fft=window_size,
            n_mels=mel_bins,
            fmin=fmin,
            fmax=fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True,
        )

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2,
        )

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        ############ modification ##################
        # Removed to get rid of dependency. We aren't doing any training

        # # Mixup on spectrogram
        # if self.training and mixup_lambda is not None:
        #     x = do_mixup(x, mixup_lambda)
        ############ end modification ##################

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        ############ modification ##################
        # Removed ReLU to get unrectified embeddings
        # x = F.relu_(self.fc1(x))
        x = self.fc1(x)
        # embedding = F.dropout(x, p=0.5, training=self.training)
        embedding = x
        # clipwise_output = torch.sigmoid(self.fc_audioset(x))
        # output_dict = {'clipwise_output': clipwise_output, 'embedding': embedding}
        output_dict = {"embedding": embedding}
        ############ end modification ##################

        return output_dict


class PannEmbedder(AudioEmbedder):
    """Link to pretrained weights: https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"""

    weights_url: str = (
        r"https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
    )

    def __init__(self):
        super().__init__()
        self.model = Cnn14(
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=527,
        )

        self.expected_sample_rate = 32000
        self.embedding_dim = 2048

    @classmethod
    def from_pretrained(cls):
        """Download/load pretrained weights and return an instance
        of the embedder with those weights loaded.
        """
        model = cls()
        state_dict = torch.hub.load_state_dict_from_url(
            cls.weights_url,
            map_location="cpu",
        )
        model.model.load_state_dict(state_dict["model"])
        model.model.eval()
        return model

    def embed(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute embeddings for a batch of audio samples.

        Args:
            audio (torch.Tensor): Tensor of shape (batch_size, 1, num_samples)

        Returns:
            torch.Tensor: Embeddings of shape (batch_size, embedding_dim)
        """

        if audio.shape[-2] != 1:
            raise ValueError(
                f"Expected audio to have one (1) channel but got {audio.shape[-2]}"
            )

        with torch.no_grad():
            output_dict = self.model(audio.squeeze(-2))
            embeddings = output_dict["embedding"]
        return embeddings
