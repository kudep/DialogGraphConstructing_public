import json
from pathlib import Path

from tqdm import tqdm


def load_multiwoz(split, path, order=None):
    data_dir = Path(path) / split
    data = []
    data_parts = list(data_dir.iterdir())
    if order:
        data_parts = [data_dir / order_item for order_item in order]
    print(f'Loading {split} part of MultiWOZ from {path}')
    for data_part in tqdm(data_parts):
        with data_part.open() as f:
            data.extend(json.load(f))
    return data
