"""Datasets and data loaders over the preprocessed frames (decisions.md §4, §7)."""
from .flare_dataset import (COARSE, FINE, MODES, N_FRAMES, FlareDataset, availability_report,  # noqa: F401
                            class_weights, default_cache_dir, filter_available, focal_alpha, load_samples, make_loaders)
