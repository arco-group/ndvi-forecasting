"""
Tools for preparing and inspecting vegetation time-series datasets.
The functions follow the MATNet style but are adapted to the agricultural context.
"""

from .cache_builder_unified import DatasetCacheBuilder, build_default_cache
from .scaler import Scaler
from .torch_dataset import CacheTimeSeriesDataset

__all__ = [
    "DatasetCacheBuilder",
    "CacheTimeSeriesDataset",
    "build_default_cache",
    "Scaler",
]
