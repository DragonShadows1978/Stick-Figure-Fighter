"""
Disk cache for generated assets.

Keys include a hash of the source files that generate the asset, so editing a
theme's code automatically invalidates its cached art.
"""
import hashlib
import inspect
import os
from typing import Callable, Iterable

import numpy as np

CACHE_DIR = os.environ.get(
    "SFF_ASSET_CACHE",
    os.path.join(os.path.expanduser("~"), ".cache", "stick-figure-fighter", "assets"))

_source_hashes = {}


def source_hash(*objs) -> str:
    """Short hash of the source files defining the given modules/functions/classes."""
    h = hashlib.sha1()
    for o in objs:
        path = inspect.getsourcefile(o) if not isinstance(o, str) else o
        if path not in _source_hashes:
            with open(path, "rb") as fh:
                _source_hashes[path] = hashlib.sha1(fh.read()).hexdigest()
        h.update(_source_hashes[path].encode())
    return h.hexdigest()[:10]


def cached(key: str, builder: Callable[[], np.ndarray], deps: Iterable = ()) -> np.ndarray:
    """
    Return builder()'s premultiplied RGBA image as uint8, cached on disk.
    `deps`: modules/functions whose source invalidates the cache when edited.
    """
    from . import raster  # noqa: F401  (raster changes also invalidate)
    tag = source_hash(raster, *deps)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
    path = os.path.join(CACHE_DIR, f"{safe}-{tag}.npz")
    try:
        with np.load(path) as data:
            return data["img"]
    except (OSError, KeyError, ValueError):
        pass
    img = builder()
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = path + ".tmp.npz"
        np.savez_compressed(tmp, img=img)
        os.replace(tmp, path)
    except OSError:
        pass
    return img
