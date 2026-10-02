"""Reproducible augmentation: one seed per sample, drawn in the main process.

albumentations 2.x and MONAI give every random transform its own generator,
seeded from the OS — `hp.seed` never reached them, so two runs with the same seed
augmented differently, and every DataLoader worker (a copy of the same generators)
drew the same augmentation sequence. Neither generator lives where it can be
saved for a resumed run either.

So the training loader's sampler draws, in the main process and from torch's
global RNG, a shuffle order and a seed per sample, and hands the dataset
`(index, seed)` pairs; the dataset reseeds its transform from that seed before
transforming. The result depends only on the torch RNG state — the same with any
number of workers, and restored exactly when a run is resumed.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, Sampler


class SeededSampler(Sampler):
    def __init__(self, n: int, shuffle: bool = True):
        self.n = n
        self.shuffle = shuffle

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(int(torch.randint(0, 2**62, (1,)).item()))
        order = torch.randperm(self.n, generator=g) if self.shuffle else torch.arange(self.n)
        seeds = torch.randint(0, 2**31 - 1, (self.n,), generator=g)
        for i, s in zip(order.tolist(), seeds.tolist()):
            yield i, s


class SeededDataset(Dataset):
    """Wraps a dataset with a `transform` so that it takes `(index, seed)` keys."""

    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            i, seed = key
            reseed(getattr(self.base, "transform", None), seed)
            return self.base[i]
        return self.base[key]


def reseed(transform, seed: int) -> None:
    if transform is None:
        return
    if hasattr(transform, "set_random_seed"):          # albumentations
        transform.set_random_seed(seed)
    elif hasattr(transform, "set_random_state"):       # MONAI
        transform.set_random_state(seed=seed)
