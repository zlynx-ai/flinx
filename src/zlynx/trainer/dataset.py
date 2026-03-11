
from dataclasses import dataclass
from typing import Callable

import grain
import datasets


@dataclass
class DatasetConfig:
    # ── source ──
    path: str | None = None                      # HF dataset name or local path
    subset: str | None = None                    # dataset config/subset name
    split: str = "train"
    eval_split: str | None = None                # e.g. "validation"
    streaming: bool = False

    # ── preprocessing ──
    preprocessing_fn: Callable | None = None
    filter_fn: Callable | None = None

    # ── shuffling / sampling ──
    shuffle: bool = True
    shuffle_seed: int = 42

    # ── performance ──
    num_workers: int = 4
    num_threads: int = 16
    prefetch_buffer_size: int = 1_000



def load_dataset(dataset, config: DatasetConfig | None = None):
    """Load a dataset and return a random-access source for grain.

    Accepts:
        str: HF dataset name or local path → loaded via datasets.load_dataset
        datasets.Dataset: used directly (supports __getitem__ + __len__)
        list: used directly as a random-access source
        datasets.IterableDataset: NOT supported (grain MapDataset needs random access)

    Returns:
        A source compatible with grain.MapDataset.source()
    """
    config = config or DatasetConfig()

    if isinstance(dataset, str):
        dataset = datasets.load_dataset(
            dataset,
            name=config.subset,
            split=config.split,
            streaming=config.streaming,
        )

    if isinstance(dataset, datasets.Dataset):
        return dataset
    elif isinstance(dataset, datasets.IterableDataset):
        raise TypeError(
            "Streaming/IterableDataset is not supported with grain MapDataset "
            "(requires random access). Use streaming=False."
        )
    elif isinstance(dataset, (list, dict)):
        return dataset
    else:
        raise TypeError(f"Unsupported dataset type: {type(dataset)}")



def process_dataset(ds, dsconfig, trconfig):
    """Build a grain pipeline from a raw dataset source.

    Returns:
        Tuple of (pipeline, num_examples) where pipeline is a grain IterDataset
        and num_examples is the number of examples in the source (for step estimation).
    """
    map_fn = lambda x: dsconfig.preprocessing_fn(x) if dsconfig.preprocessing_fn is not None else x
    fil_fn = lambda x: dsconfig.filter_fn(x) if dsconfig.filter_fn is not None else True
    read_opts = grain.ReadOptions(
        num_threads=dsconfig.num_threads,
        prefetch_buffer_size=dsconfig.prefetch_buffer_size,
    )

    source = load_dataset(ds, dsconfig)
    num_examples = len(source) if hasattr(source, '__len__') else None

    pipeline = grain.MapDataset.source(source)

    if dsconfig.shuffle:
        pipeline = pipeline.shuffle(seed=dsconfig.shuffle_seed)

    pipeline = (
        pipeline
        .map(map_fn)
        .to_iter_dataset(read_opts)
        .filter(fil_fn)
        .batch(batch_size=trconfig.batch_size)
    )

    return pipeline, num_examples