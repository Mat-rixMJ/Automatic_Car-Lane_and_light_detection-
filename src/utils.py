"""Shared utilities for the CarLaneI project."""

import json
import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"


def ensure_dirs():
    """Create necessary directories if they don't exist."""
    for d in [DATA_DIR, MODELS_DIR, OUTPUT_DIR,
              DATA_DIR / "gtsrb", DATA_DIR / "traffic_lights", DATA_DIR / "indian"]:
        d.mkdir(parents=True, exist_ok=True)


def load_config():
    """Load pipeline configuration."""
    config_path = CONFIG_DIR / "pipeline_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_sign_config(region="german"):
    """Load traffic sign class configuration."""
    config_path = CONFIG_DIR / f"{region}_signs.json"
    with open(config_path, "r") as f:
        return json.load(f)


def get_class_names(region="german"):
    """Get ordered list of class names for a region."""
    config = load_sign_config(region)
    classes = config["classes"]
    return [classes[str(i)] for i in range(len(classes))]
