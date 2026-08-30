"""Configuration loading.

One YAML file, one dataclass tree, loaded once at startup. No settings are
read from the environment except the two that legitimately differ per machine:
the config file location and the data directory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


@dataclass
class StorageConfig:
    data_dir: Path = REPO_ROOT / "data"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def catalogue_path(self) -> Path:
        return self.data_dir / "localscholar.db"


@dataclass
class IngestionConfig:
    max_upload_mb: int = 50

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@dataclass
class Config:
    storage: StorageConfig = field(default_factory=StorageConfig)
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)

    def ensure_directories(self) -> None:
        self.storage.uploads_dir.mkdir(parents=True, exist_ok=True)


def _resolve(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from YAML, falling back to dataclass defaults."""
    config_path = Path(path or os.environ.get("LOCALSCHOLAR_CONFIG") or DEFAULT_CONFIG_PATH)
    raw: dict[str, Any] = {}
    if config_path.exists():
        raw = yaml.safe_load(config_path.read_text()) or {}

    storage_raw = raw.get("storage", {}) or {}
    data_dir = os.environ.get("LOCALSCHOLAR_DATA_DIR") or storage_raw.get("data_dir")
    storage = StorageConfig(
        data_dir=_resolve(data_dir) if data_dir else StorageConfig().data_dir
    )

    ingestion_raw = raw.get("ingestion", {}) or {}
    ingestion = IngestionConfig(
        max_upload_mb=int(ingestion_raw.get("max_upload_mb", IngestionConfig().max_upload_mb))
    )

    return Config(storage=storage, ingestion=ingestion)
