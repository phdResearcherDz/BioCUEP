"""
Root configuration file for BioCUEP.
Preserves backwards compatibility with legacy stage scripts while providing absolute path resolution.
"""

import os
from pathlib import Path

# Absolute path to the repository root directory
PROJECT_ROOT = Path(__file__).resolve().parent
root_folder = str(PROJECT_ROOT) + "/"

# Relative fallback if required by certain legacy subfolders
legacy_root_folder = str(PROJECT_ROOT)

def get_root_path() -> Path:
    return PROJECT_ROOT
