from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot encode {type(value).__name__}")

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False) + "\n")

def load_json(path):
    return json.loads(Path(path).read_text())

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=json_default,
                                    allow_nan=False).encode()).hexdigest()

def normalize_probabilities(values):
    values = np.asarray(values, dtype=float)
    values = np.maximum(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0)
    total = values.sum(axis=-1, keepdims=True)
    return np.divide(values, total, out=np.full_like(values, 1 / values.shape[-1]),
                     where=total > 0)

def entropy(probabilities):
    probabilities = normalize_probabilities(probabilities)
    return -(probabilities * np.log(np.clip(probabilities, 1e-12, 1))).sum(axis=-1)
