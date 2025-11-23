from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHANPY_ROOT = PROJECT_ROOT / "third_party" / "chan.py"

if str(CHANPY_ROOT) not in sys.path:
    sys.path.insert(0, str(CHANPY_ROOT))

from ChanConfig import CChanConfig  # type: ignore

DEFAULT_CHAN_CONFIG_PATH = PROJECT_ROOT / "config" / "chan.yaml"


def load_chan_raw_config(path: Optional[str | os.PathLike[str]] = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path is not None else DEFAULT_CHAN_CONFIG_PATH
    import yaml  # type: ignore
    if not cfg_path.exists():
        raise FileNotFoundError(str(cfg_path))
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "chan" not in data:
        raise ValueError("missing top-level key 'chan'")
    if not isinstance(data["chan"], dict):
        raise TypeError("config['chan'] must be a dict")
    return data["chan"]


def build_cchan_config(
    path: Optional[str | os.PathLike[str]] = None,
    extra_overrides: Optional[Dict[str, Any]] = None,
) -> CChanConfig:
    cfg_path = Path(path) if path is not None else DEFAULT_CHAN_CONFIG_PATH
    conf: Dict[str, Any]
    if cfg_path.exists():
        conf = load_chan_raw_config(str(cfg_path))
    else:
        conf = {}
    if extra_overrides:
        conf = {**conf, **extra_overrides}
    return CChanConfig(conf)


if __name__ == "__main__":
    cfg = build_cchan_config()
    print("chan.trigger_step=", cfg.trigger_step)
    cfg2 = build_cchan_config(extra_overrides={"trigger_step": True, "skip_step": 1})
    print("override.trigger_step=", cfg2.trigger_step, "skip_step=", cfg2.skip_step)