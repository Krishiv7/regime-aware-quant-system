import os
import pandas as pd
from pathlib import Path
from typing import Optional

CACHE_DIR = Path("data")
RAW_DIR = CACHE_DIR / "raw"
PROCESSED_DIR = CACHE_DIR / "processed"

def ensure_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def get_cache_path(identifier: str, stage: str = "raw") -> Path:
    ensure_dirs()
    if stage == "raw":
        return RAW_DIR / f"{identifier}.parquet"
    else:
        return PROCESSED_DIR / f"{identifier}.parquet"

def load_from_cache(identifier: str, stage: str = "raw") -> Optional[pd.DataFrame]:
    path = get_cache_path(identifier, stage)
    if path.exists():
        return pd.read_parquet(path)
    return None

def save_to_cache(df: pd.DataFrame, identifier: str, stage: str = "raw") -> None:
    path = get_cache_path(identifier, stage)
    df.to_parquet(path, index=True)
