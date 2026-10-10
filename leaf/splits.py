"""Candidate search and the three-term posterior split rule."""
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
import json
import numpy as np
import pandas as pd
from .utils import entropy, json_default, normalize_probabilities

@dataclass(frozen=True)
class SplitCandidate:
    feature: str
    kind: str
    threshold: float | None = None
    categories_left: tuple[str, ...] = ()

def categorical_values(series):
    return series.astype("string").fillna("__MISSING__").str.strip().str.lower()

def apply_candidate(df, candidate):
    if candidate.kind == "numeric":
        values = pd.to_numeric(df[candidate.feature], errors="coerce")
        left = values.fillna(-np.inf).to_numpy() <= candidate.threshold
    else:
        left = categorical_values(df[candidate.feature]).isin(
            candidate.categories_left).to_numpy(dtype=bool)
    return left, ~left

def candidate_rule(candidate):
    if candidate.kind == "numeric":
        return f"{candidate.feature} <= {candidate.threshold:.12g} (missing goes left)"
    return f"{candidate.feature} in {list(candidate.categories_left)}"

def generate_candidates(df, features, numeric_cols, config, metadata=None):
    per_feature = []
    orders = (metadata or {}).get("__ordinal_orders__", {})
    generic_orders = [
        ["low", "med", "high"], ["small", "med", "big"],
        ["poor", "fair", "good", "excellent"], ["bad", "average", "good"],
    ]
    for feature in features:
        candidates = []
        if feature in numeric_cols:
            values = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if len(np.unique(values)) < 2:
                continue
            thresholds = np.unique(np.quantile(
                values, np.linspace(.15, .85, config.max_candidates_per_feature)))
            candidates = [SplitCandidate(feature, "numeric", float(t)) for t in thresholds]
        else:
            levels = categorical_values(df[feature]).value_counts().index.tolist()
            if len(levels) < 2:
                continue
            order = orders.get(feature, []) if isinstance(orders, dict) else []
            if not order:
                order = next((o for o in generic_orders if set(levels).issubset(o)), [])
            if order:
                order = [str(v).strip().lower() for v in order]
                # Ordinal partitions use contiguous prefixes.
                ordered = [v for v in order if v in levels]
                for index in range(1, len(ordered)):
                    candidates.append(SplitCandidate(
                        feature, "categorical", categories_left=tuple(ordered[:index])))
            elif len(levels) <= 6:
                for count in range(1, len(levels) // 2 + 1):
                    for group in combinations(levels, count):
                        if count * 2 == len(levels) and levels[0] not in group:
                            continue
                        candidates.append(SplitCandidate(
                            feature, "categorical", categories_left=tuple(group)))
            else:
                candidates = [SplitCandidate(feature, "categorical", categories_left=(v,))
                              for v in levels[:config.max_candidates_per_feature]]
        per_feature.append(candidates[:config.max_candidates_per_feature])
    # Interleave features to avoid spending the node budget on one column.
    output = []
    for index in range(config.max_candidates_per_feature):
        for candidates in per_feature:
            if index < len(candidates):
                output.append(candidates[index])
                if len(output) >= config.max_candidates_per_node:
                    return output
    return output

def valid_candidates(candidates, df, labeled_mask, config):
    valid = []
    n_labeled = int(labeled_mask.sum())
    for candidate in candidates:
        left, right = apply_candidate(df, candidate)
        if min(int(left.sum()), int(right.sum())) < config.min_total_child:
            continue
        if n_labeled and min(int((left & labeled_mask).sum()),
                             int((right & labeled_mask).sum())) < config.min_labeled_child:
            continue
        valid.append(candidate)
    return valid

def impurity(values, task, n_classes=None):
    if len(values) == 0:
        return 0.0
    if task == "classification":
        counts = np.bincount(np.asarray(values, dtype=int), minlength=n_classes)
        return float(entropy(counts))
    return float(np.var(np.asarray(values, dtype=float)))

def split_utilities(task, labels, teacher_predictions, labeled_mask, left, n_classes=None):
    """Appendix definitions: label gain, teacher gain, routing stability."""
    right = ~left
    labels = np.asarray(labels)
    labeled_mask = np.asarray(labeled_mask, dtype=bool)
    n_all, n_lab = len(left), int(labeled_mask.sum())
    if n_all == 0:
        return np.zeros(3)
    lab_left = left[labeled_mask]
    q_lab = impurity(labels, task, n_classes)
    if n_lab:
        for mask in (lab_left, ~lab_left):
            q_lab -= mask.mean() * impurity(labels[mask], task, n_classes)
        q_str = 1 - .5 * (
            abs(float(lab_left.mean()) - float(left.mean())) +
            abs(float((~lab_left).mean()) - float(right.mean()))
        )
    else:
        q_lab, q_str = 0.0, 0.0
    teacher = np.asarray(teacher_predictions, dtype=float)
    def teacher_impurity(pred):
        if len(pred) == 0:
            return 0.0
        if task == "classification":
            return float(entropy(np.mean(pred, axis=0)))
        return float(np.var(pred))
    q_tea = teacher_impurity(teacher)
    for mask in (left, right):
        q_tea -= mask.mean() * teacher_impurity(teacher[mask])
    # Floating point roundoff can make a mathematically nonnegative gain negative.
    return np.asarray([max(0.0, q_lab), max(0.0, q_tea), max(0.0, q_str)])

def posterior_probabilities(priors, utilities, config):
    priors = normalize_probabilities(priors)
    utility = np.asarray(utilities) @ np.asarray(
        [config.lambda_lab, config.lambda_tea, config.lambda_str])
    with np.errstate(divide="ignore"):
        logits = np.log(priors) + config.eta * utility
    weights = np.exp(logits - np.max(logits))
    return normalize_probabilities(weights)

def node_summary(df):
    output = {}
    for column in df:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").replace(
                [np.inf, -np.inf], np.nan).dropna()
            output[column] = {
                "type": "numeric", "missing": float(series.isna().mean()),
                "quantiles": numeric.quantile([0, .25, .5, .75, 1]).tolist(),
            } if len(numeric) else {"type": "numeric", "all_missing": True}
        else:
            output[column] = {
                "type": "categorical",
                "counts": categorical_values(series).value_counts().head(8).to_dict(),
            }
    return output

def semantic_priors(llm, candidates, df, labeled_examples, target, task,
                    labels, path, metadata):
    uniform = np.full(len(candidates), 1 / len(candidates))
    context = {
        "task_type": task, "target_column": target, "class_labels": labels,
        "task_description": metadata.get("__dataset_description__", ""),
        "target_description": metadata.get("__target_description__", ""),
        "current_path": path, "feature_meanings": metadata,
        "few_shot_labeled_examples": labeled_examples.to_dict(orient="records"),
        "node_summary": node_summary(df),
        "candidate_splits": [{"id": i, "rule": candidate_rule(c)}
                             for i, c in enumerate(candidates)],
    }
    prompt = (
        "You are assigning an LLM prior over candidate splits for a few-shot tabular forest.\n"
        "Use semantic plausibility, the current path, few labeled examples, and the "
        "labeled-unlabeled feature distribution. Prefer coherent subpopulations likely "
        "to generalize beyond the scarce labels. Numeric thresholds must lie inside "
        "the observed distribution; categorical values must match observed categories; "
        "ordinal categories should use contiguous threshold-like groups.\n"
        "Score only the existing candidates; do not propose or modify them. "
        "Return JSON only with key scores. Each item has id, score in [0,1], and rationale.\n"
        + json.dumps(context, default=json_default, allow_nan=False)
    )
    raw = llm.generate_json(prompt, default={})
    items = raw.get("scores", []) if isinstance(raw, dict) else []
    parsed = {}
    for item in items if isinstance(items, list) else []:
        try:
            index = int(item["id"])
            score = float(item["score"])
            if index in parsed or not 0 <= index < len(candidates) or not 0 <= score <= 1:
                return uniform
            parsed[index] = score
        except (KeyError, TypeError, ValueError):
            return uniform
    if len(parsed) != len(candidates):
        return uniform
    return normalize_probabilities([parsed[i] for i in range(len(candidates))])
