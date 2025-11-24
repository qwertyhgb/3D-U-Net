from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """加载 YAML 配置为字典。"""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str | Path) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True)
