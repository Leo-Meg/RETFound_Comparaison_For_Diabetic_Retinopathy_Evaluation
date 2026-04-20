"""Device selection helpers."""

from __future__ import annotations

import torch


def choose_device(preferred: str = "auto") -> torch.device:
    """Choose a torch device.

    Priority in auto mode: MPS, CUDA, CPU.
    """

    preferred = preferred.lower()
    if preferred != "auto":
        return torch.device(preferred)

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
