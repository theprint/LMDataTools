# datacore/config/loader.py
"""
load_tool_config - Shared configuration loading utility.

Merges a tool's DEFAULT_CONFIG dict with values found in config.json in the
current working directory.  Keys are normalised to UPPER_CASE so both
``{"dataset_name": ...}`` and ``{"DATASET_NAME": ...}`` from the config file
work identically.

Usage
-----
    from datacore.config.loader import load_tool_config

    DEFAULT_CONFIG = {
        "DATASET_NAME": "my-dataset",
        "SAVE_INTERVAL": 250,
    }
    cfg = load_tool_config(DEFAULT_CONFIG, tool_name="mytool")
    DATASET_NAME = cfg["DATASET_NAME"]
"""

import os
import json


def load_tool_config(
    defaults: dict,
    config_path: str = None,
    tool_name: str = None,
) -> dict:
    """
    Load tool configuration by merging *defaults* with config.json overrides.

    Parameters
    ----------
    defaults:
        Default configuration values (keys should be UPPER_CASE).
        The original dict is **not** mutated; a copy is returned.
    config_path:
        Path to a specific config file.  Defaults to ``config.json`` in the
        current working directory.
    tool_name:
        Tool name used in log messages (e.g. ``"databird"``).

    Returns
    -------
    dict
        Merged configuration dict with all keys in UPPER_CASE.
    """
    label = tool_name or "tool"
    cfg = dict(defaults)  # shallow copy — caller's defaults are not mutated

    path = config_path or os.path.join(os.getcwd(), "config.json")
    if os.path.exists(path):
        print(f"[{label}] Loading configuration from {path}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                user_config = json.load(f)
            for key, value in user_config.items():
                cfg[key.upper()] = value
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[{label}] Warning: Could not load config.json — {exc}")

    return cfg
