"""Paths and small shared settings."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(os.environ.get("DTAR_ROOT", Path(__file__).resolve().parents[1]))
DATA = ROOT / "data"
DATA_SCALE = ROOT / "data_scale"
GOLD = ROOT / "gold"
CONFIG = ROOT / "config"
RESULTS = ROOT / "results"
# Serverless hosts (Vercel) have a read-only project folder; caches go to /tmp there.
CACHE = Path("/tmp/.cache") if os.environ.get("VERCEL") else ROOT / ".cache"
SEED = 42

for _p in (RESULTS, CACHE):
    try:
        _p.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass


@lru_cache(maxsize=None)
def load_yaml(name: str) -> dict:
    with open(CONFIG / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_data_dir(source: str = "local", drive_path: str = "/content/drive/MyDrive/Domain_Text_Analysis_Retrieval/data") -> Path:
    """Return the folder that holds D01-D30.

    source="local"  -> ./data next to this package
    source="drive"  -> Google Drive (Colab). Mounts the drive if needed.
    """
    if source == "drive":
        try:
            from google.colab import drive  # type: ignore

            drive.mount("/content/drive")
        except ImportError as exc:  # not in Colab
            raise RuntimeError("source='drive' needs Google Colab") from exc
        return Path(drive_path)
    return DATA
