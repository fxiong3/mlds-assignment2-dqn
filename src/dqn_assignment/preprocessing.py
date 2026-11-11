"""Observation preprocessing for vector and Atari inputs."""

from __future__ import annotations

import numpy as np
import torch

MSPACMAN_COLOR_SUM = 210 + 164 + 74


def preprocess_mspacman(observation: np.ndarray) -> np.ndarray:
    """Apply the exact crop, grayscale, and normalization specified in the PDF.

    The returned array is channel-first so it can be passed directly to PyTorch.
    Keeping int8 observations in replay reduces memory by 4x versus float32. With
    the four-frame MsPacman preset, 50,000 state/next-state pairs use roughly 2.8 GB.
    """

    if observation.shape != (210, 160, 3):
        raise ValueError(
            f"MsPacman observation must have shape (210, 160, 3); received {observation.shape}"
        )
    image = observation[1:176:2, ::2].sum(axis=2)
    image[image == MSPACMAN_COLOR_SUM] = 0
    image = (image // 3 - 128).astype(np.int8)
    return image[np.newaxis, ...]


def prepare_observations(observations: torch.Tensor) -> torch.Tensor:
    """Convert replay tensors to network-ready float32 values."""

    if observations.dtype == torch.int8:
        return observations.to(torch.float32).div(128.0)
    if observations.dtype == torch.uint8:
        return observations.to(torch.float32).div(255.0)
    return observations.to(torch.float32)
