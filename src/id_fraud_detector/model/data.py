"""Photos for the classifier: one reduced copy of each photo, kept in a single local file.

Decoding a 2 MB PNG on every epoch would make training many times slower, so each photo is
reduced once to the input size of the network and stored in <output>/cache/<side>_<W>x<H>.npy
(uint8, N x H x W x 3) with an index of the people next to it. Training reads that file.

The whole photo is used, without cropping the card (docs/plan.md §0). It is scaled until it
covers the input size and the excess is cut evenly from both sides: no bars are added, because
a bar would tell the network the shape of the camera frame, and the camera depends on the
channel.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def reduced(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """The photo as an H x W x 3 uint8 array of `size` (width, height): scaled to cover it,
    centered, excess cut."""
    return np.asarray(ImageOps.fit(image.convert("RGB"), size, Image.LANCZOS, centering=(0.5, 0.5)))


def load_reduced(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        return reduced(image, size)


def cache_paths(out: Path, side: str, size: tuple[int, int], tag: str = "") -> tuple[Path, Path]:
    name = f"{side}_{size[0]}x{size[1]}{tag}"
    return out / "cache" / f"{name}.npy", out / "cache" / f"{name}.index.csv"


def build_cache(out: Path, side: str, size: tuple[int, int], anon_ids: list[str], paths: list[Path],
                tag: str = "", workers: int = 8) -> np.ndarray:
    """Reduced photos of `anon_ids`, in that order, as a read-only memory map. The file is
    reused when it already holds exactly those people in that order; otherwise it is rebuilt.
    `tag` names a separate file (a trial with a few people must not replace the full one)."""
    array_path, index_path = cache_paths(out, side, size, tag)
    shape = (len(anon_ids), size[1], size[0], 3)
    if array_path.exists() and index_path.exists():
        if list(pd.read_csv(index_path)["anon_id"]) == list(anon_ids):
            cached = np.load(array_path, mmap_mode="r")
            if cached.shape == shape:
                return cached
    array_path.parent.mkdir(parents=True, exist_ok=True)
    partial = array_path.with_suffix(".npy.part")
    array = np.lib.format.open_memmap(partial, mode="w+", dtype=np.uint8, shape=shape)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        photos = pool.map(lambda path: load_reduced(path, size), paths)
        for i, photo in enumerate(tqdm(photos, total=len(paths), desc=f"   reducing {side} photos")):
            array[i] = photo
    array.flush()
    del array
    partial.replace(array_path)
    pd.DataFrame({"anon_id": anon_ids}).to_csv(index_path, index=False)
    return np.load(array_path, mmap_mode="r")
