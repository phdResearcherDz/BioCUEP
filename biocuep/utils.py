"""
Utility functions and safe fallbacks for BioCUEP.
"""

import json
import logging
from pathlib import Path
from typing import Any, Iterable, List, Dict

# Safe tqdm fallback
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable: Iterable = None, *args, **kwargs):
        return iterable if iterable is not None else []


def setup_logger(name: str = "biocuep", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def load_json(filepath: Path | str) -> Any:
    """Safe JSON loader."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, filepath: Path | str, indent: int = 4) -> None:
    """Safe JSON saver ensuring directory exists."""
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_dataset_file(filepath: Path | str) -> List[Dict[str, Any]]:
    """
    Robust dataset loader supporting both standard JSON array and JSONL (BLURB format).
    Normalizes field names:
      'sentence1' -> 'question'
      'sentence2' -> 'context'
      'label' -> 'answer'
    """
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(p, "r", encoding="utf-8") as f:
        content = f.read().strip()

    items = []
    # Try standard JSON list first
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict):
            if "questions" in parsed and isinstance(parsed["questions"], list):
                items = parsed["questions"]
            else:
                items = [parsed]
    except json.JSONDecodeError:
        # Fallback to JSON Lines (BLURB format used in Data/)
        for line in content.splitlines():
            line_str = line.strip()
            if line_str:
                try:
                    items.append(json.loads(line_str))
                except json.JSONDecodeError:
                    continue

    normalized = []
    for it in items:
        if not isinstance(it, dict):
            continue
        item = dict(it)
        # Normalize BLURB field names
        if "sentence1" in item and "question" not in item:
            item["question"] = item["sentence1"]
        if "sentence2" in item and "context" not in item:
            item["context"] = item["sentence2"]
        if "label" in item and "answer" not in item:
            item["answer"] = item["label"]

        normalized.append(item)

    return normalized
