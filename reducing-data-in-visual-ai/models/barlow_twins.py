"""Barlow Twins on a randomly-initialised ViT-Tiny/8 backbone.

The encoder is built from `shared_config.VIT_TINY_8_KWARGS` so this method
shares its backbone definition with every other method in the group.
"""
import os
import sys

import torch
import torch.nn as nn
from lightly.models.modules import BarlowTwinsProjectionHead
from transformers import ViTConfig, ViTModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_config import VIT_TINY_8_KWARGS

ENCODER_DIM = VIT_TINY_8_KWARGS["hidden_size"]  # 192
PROJECTOR_HIDDEN = 1024
PROJECTOR_OUT = 1024


def build_vit_tiny_encoder() -> ViTModel:
    config = ViTConfig(**VIT_TINY_8_KWARGS)
    return ViTModel(config, add_pooling_layer=False)


class BarlowTwinsViT(nn.Module):
    def __init__(
        self,
        projector_hidden: int = PROJECTOR_HIDDEN,
        projector_out: int = PROJECTOR_OUT,
    ):
        super().__init__()
        self.backbone = build_vit_tiny_encoder()
        self.projector = BarlowTwinsProjectionHead(
            input_dim=ENCODER_DIM,
            hidden_dim=projector_hidden,
            output_dim=projector_out,
        )

    def embed(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.projector(self.embed(pixel_values))
