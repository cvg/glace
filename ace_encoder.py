# Copyright © Niantic, Inc. 2022.

from dataclasses import dataclass, field
from typing import Optional, Type

import torch
import torch.nn.functional as F
from torch import nn

from scrstudio.encoders.base_encoder import Encoder, EncoderConfig, PreprocessConfig


@dataclass
class ACEEncoderConfig(EncoderConfig):
    """Configuration for Encoder instantiation"""

    _target: Type = field(default_factory=lambda: ACEEncoder)
    """target class to instantiate"""
    
    out_channels: int = 512

    ckpt_path: Optional[str] = "ace_encoder_pretrained.pt"



class ACEEncoder(Encoder):
    """
    FCN encoder, used to extract features from the input images.

    The number of output channels is configurable, the default used in the paper is 512.
    """
    OUTPUT_SUBSAMPLE = 8
    def __init__(self, config: ACEEncoderConfig, **kwargs):
        super().__init__(config)
        self.preprocess = PreprocessConfig(mean=0.4, std=0.25, grayscale=True, use_half=True, size_multiple=8)

        self.out_channels = config.out_channels
        ckpt_path = config.ckpt_path
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)

        self.res1_conv1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.res1_conv2 = nn.Conv2d(256, 256, 1, 1, 0)
        self.res1_conv3 = nn.Conv2d(256, 256, 3, 1, 1)

        self.res2_conv1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.res2_conv2 = nn.Conv2d(512, 512, 1, 1, 0)
        self.res2_conv3 = nn.Conv2d(512, self.out_channels, 3, 1, 1)

        self.res2_skip = nn.Conv2d(256, self.out_channels, 1, 1, 0)

        if ckpt_path is not None:
            self.load_state_dict(torch.load(ckpt_path, map_location='cpu'))

    def forward(self, data,det=False):
        x = data["image"]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        res = F.relu(self.conv4(x))

        x = F.relu(self.res1_conv1(res))
        x = F.relu(self.res1_conv2(x))
        x = F.relu(self.res1_conv3(x))

        res = res + x

        x = F.relu(self.res2_conv1(res))
        x = F.relu(self.res2_conv2(x))
        x = F.relu(self.res2_conv3(x))

        x = self.res2_skip(res) + x

        return {
            "features": x
        }

