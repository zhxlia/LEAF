"""Benchmark loading and the existing few-shot sampling protocol."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import CLASSIFICATION_DATASETS, REGRESSION_DATASETS
from .utils import load_json

@dataclass
class PreparedDataset:
    labeled: pd.DataFrame
    unlabeled: pd.DataFrame
    test: pd.DataFrame
    target: str
    labels: list | None
    numeric_cols: list[str]
    categorical_cols: list[str]
    metadata: dict

def read_dataset(data_root, dataset):
    root = Path(data_root)
    candidates = [root / f"{dataset}.csv", root / dataset / f"{dataset}.csv"]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Dataset {dataset}.csv is missing under {root}")
    df = pd.read_csv(path)
    if not df.columns.is_unique or any(str(c).startswith("__") for c in df.columns):
        raise ValueError("Columns must be unique and must not use reserved '__' names")
    if len(df.columns) < 2 or df.iloc[:, -1].isna().any():
        raise ValueError("Expected features and a non-missing target in the last column")
    metadata_path = path.with_name(f"{dataset}-metadata.json")
    metadata = load_json(metadata_path) if metadata_path.exists() else {}
    if not isinstance(metadata, dict):
        raise ValueError("Metadata must be a JSON object")
    return df, metadata

def feature_types(df):
    numeric = [c for c in df if pd.api.types.is_numeric_dtype(df[c])]
    return numeric, [c for c in df if c not in numeric]

def prepare_dataset(df, metadata, task, shot, seed):
    target = df.columns[-1]
    if task == "regression":
        df = df.copy()
        df[target] = pd.to_numeric(df[target], errors="raise")
        if not np.isfinite(df[target]).all():
            raise ValueError("Regression targets must be finite")
    train, test = train_test_split(
        df, test_size=0.2, random_state=seed,
        stratify=df[target] if task == "classification" else None,
    )
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    if shot < 1 or shot > len(train):
        raise ValueError("shot must lie between 1 and the training-pool size")
    if task == "classification":
        labels = sorted(train[target].unique().tolist())
        if shot < len(labels):
            raise ValueError("Total shot budget must include at least one row per class")
        labeled, unlabeled = sample_fewshot_classification(train, target, shot, seed)
    else:
        labels = None
        labeled, unlabeled = sample_fewshot_regression(train, target, shot, seed)
    numeric, categorical = feature_types(train.drop(columns=target))
    # Hidden unlabeled targets never enter feature construction, teachers, or splits.
    unlabeled = unlabeled.drop(columns=target)
    return PreparedDataset(labeled, unlabeled, test, target, labels,
                           numeric, categorical, metadata)

def validate_datasets(task, datasets):
    allowed = CLASSIFICATION_DATASETS if task == "classification" else REGRESSION_DATASETS
    unknown = set(datasets) - set(allowed)
    if unknown:
        raise ValueError(f"Datasets outside the paper benchmark: {sorted(unknown)}")

def sample_fewshot_classification(df: pd.DataFrame, target_col: str, shot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    y = df[target_col].astype('object')
    classes = list(pd.Index(y.dropna().unique().tolist()))
    if len(classes) == 0:
        raise ValueError('Empty classification target')
    base = max(1, shot // len(classes))
    remainder = max(0, shot - base * len(classes))
    labeled_indices: list[int] = []
    for (i, cls) in enumerate(classes):
        idx = np.where(y.to_numpy() == cls)[0]
        rng.shuffle(idx)
        take = min(len(idx), base + (1 if i < remainder else 0))
        labeled_indices.extend(idx[:take].tolist())
    labeled_indices = sorted(set(labeled_indices))
    if len(labeled_indices) < min(shot, len(df)):
        remaining = [i for i in range(len(df)) if i not in labeled_indices]
        rng.shuffle(remaining)
        need = min(len(df), shot) - len(labeled_indices)
        labeled_indices.extend(remaining[:need])
    labeled = df.iloc[sorted(labeled_indices)].reset_index(drop=True)
    unlabeled = df.drop(df.index[sorted(labeled_indices)]).reset_index(drop=True)
    return (labeled, unlabeled)

def sample_fewshot_regression(df: pd.DataFrame, target_col: str, shot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    shot = min(max(int(shot), 0), len(df))
    if shot == 0:
        return (df.iloc[[]].copy().reset_index(drop=True), df.reset_index(drop=True))
    y = pd.to_numeric(df[target_col], errors='coerce')
    y_arr = y.to_numpy(dtype=float)
    finite_idx = np.where(np.isfinite(y_arr))[0]
    if len(finite_idx) == 0:
        labeled_indices = sorted(rng.choice(np.arange(len(df)), size=shot, replace=False).tolist())
        labeled = df.iloc[labeled_indices].reset_index(drop=True)
        unlabeled = df.drop(df.index[labeled_indices]).reset_index(drop=True)
        return (labeled, unlabeled)
    labeled_indices: list[int] = []
    finite_count = int(len(finite_idx))
    q = min(shot, finite_count)
    q = max(1, min(q, shot, finite_count))
    groups = pd.Series(pd.NA, index=df.index, dtype='object')
    try:
        ranks = y.loc[finite_idx].rank(method='first')
        codes = pd.qcut(ranks, q=q, labels=False, duplicates='drop')
        groups.loc[finite_idx] = pd.Series(codes, index=finite_idx).astype('object')
    except Exception:
        groups.loc[finite_idx] = 0
    uniq = [g for g in pd.Index(groups.loc[finite_idx].dropna().unique().tolist()).tolist()]
    if not uniq:
        uniq = [0]
        groups.loc[finite_idx] = 0
    rng.shuffle(uniq)
    base = shot // len(uniq)
    rem = shot % len(uniq)
    for (i, g) in enumerate(uniq):
        idx = np.where(groups.to_numpy() == g)[0]
        idx = idx[np.isfinite(y_arr[idx])]
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        take = min(len(idx), base + (1 if i < rem else 0))
        labeled_indices.extend(idx[:take].tolist())
    labeled_indices = sorted(set((int(i) for i in labeled_indices)))
    if len(labeled_indices) < shot:
        remaining = [int(i) for i in finite_idx.tolist() if int(i) not in labeled_indices]
        rng.shuffle(remaining)
        labeled_indices.extend(remaining[:shot - len(labeled_indices)])
    if len(labeled_indices) < shot:
        remaining = [i for i in range(len(df)) if i not in labeled_indices]
        rng.shuffle(remaining)
        labeled_indices.extend(remaining[:shot - len(labeled_indices)])
    labeled_indices = sorted(set(labeled_indices))[:shot]
    labeled = df.iloc[labeled_indices].reset_index(drop=True)
    unlabeled = df.drop(df.index[labeled_indices]).reset_index(drop=True)
    return (labeled, unlabeled)
