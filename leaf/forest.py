"""Posterior-guided trees with static empirical/teacher leaf estimates."""
from __future__ import annotations
from dataclasses import dataclass
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from .config import LEAFConfig
from .features import apply_feature_specs
from .splits import (
    apply_candidate, candidate_rule, generate_candidates, posterior_probabilities,
    semantic_priors, split_utilities, valid_candidates,
)
from .utils import normalize_probabilities

@dataclass
class TreeNode:
    split: object | None = None
    prediction: object | None = None
    left: TreeNode | None = None
    right: TreeNode | None = None

def hybrid_leaf(task, labeled_values, teacher_predictions, config, n_classes=None):
    n_labeled = len(labeled_values)
    alpha = n_labeled / (n_labeled + config.kappa)
    if task == "classification":
        counts = np.bincount(np.asarray(labeled_values, dtype=int), minlength=n_classes)
        empirical = normalize_probabilities(counts + config.gamma)
        teacher = normalize_probabilities(np.mean(teacher_predictions, axis=0))
        return normalize_probabilities(alpha * empirical + (1 - alpha) * teacher)
    teacher = float(np.mean(teacher_predictions))
    empirical = float(np.mean(labeled_values)) if n_labeled else teacher
    return alpha * empirical + (1 - alpha) * teacher

class LEAFForest:
    def __init__(self, task, target, labels=None, config=None, seed=0):
        if task not in {"classification", "regression"}:
            raise ValueError("task must be classification or regression")
        self.task = task
        self.target = target
        self.labels = labels
        self.config = config or LEAFConfig()
        self.seed = seed
        self.trees = []
        self.feature_specs = []
        self.feature_names = []

    def fit(self, labeled, unlabeled, teacher_predictions, llm, feature_specs, metadata):
        config = self.config
        if len(labeled) == 0:
            raise ValueError("At least one labeled row is required")
        self.feature_specs = list(feature_specs)
        raw_columns = [c for c in labeled if c != self.target]
        if set(unlabeled.columns) != set(raw_columns):
            raise ValueError("Unlabeled input must contain only raw feature columns")
        labeled_x = apply_feature_specs(labeled[raw_columns], self.feature_specs)
        unlabeled_x = apply_feature_specs(unlabeled[raw_columns], self.feature_specs)
        self.feature_names = labeled_x.columns.tolist()
        numeric_cols = [c for c in labeled_x if pd.api.types.is_numeric_dtype(labeled_x[c])]
        teacher_predictions = np.asarray(teacher_predictions, dtype=float)
        if len(teacher_predictions) != len(labeled) + len(unlabeled):
            raise ValueError("Teacher predictions must align with labeled then unlabeled rows")
        if not np.isfinite(teacher_predictions).all():
            raise ValueError("Teacher predictions must be finite")
        if self.task == "classification":
            label_index = {label: i for i, label in enumerate(self.labels)}
            y = np.asarray([label_index[label] for label in labeled[self.target]], dtype=int)
            n_classes = len(self.labels)
        else:
            y = labeled[self.target].to_numpy(dtype=float)
            n_classes = None
        metadata = dict(metadata)
        for spec in feature_specs:
            metadata[spec.name] = f"{spec.op} of {', '.join(spec.cols)}"
        rng = np.random.default_rng(self.seed)
        self.trees = []
        for tree_index in range(config.n_estimators):
            # Preserve the original protocol: bootstrap labeled rows, retain
            # all unlabeled training rows, and align cached teacher predictions.
            sample = rng.choice(len(labeled), size=len(labeled), replace=True)
            df = pd.concat([labeled_x.iloc[sample], unlabeled_x], ignore_index=True)
            labeled_mask = np.arange(len(df)) < len(sample)
            targets = np.zeros(len(df), dtype=int if self.task == "classification" else float)
            targets[:len(sample)] = y[sample]
            raw_y = np.empty(len(df), dtype=object)
            raw_y[:len(sample)] = labeled[self.target].iloc[sample].to_numpy()
            raw_y[len(sample):] = None
            prediction_indices = np.concatenate([
                sample, np.arange(len(labeled), len(labeled) + len(unlabeled)),
            ])
            predictions = teacher_predictions[prediction_indices]
            tree_rng = np.random.default_rng(self.seed + tree_index + 13)

            def build(indices, depth, path):
                node_df = df.iloc[indices].reset_index(drop=True)
                node_mask = labeled_mask[indices]
                node_y = targets[indices][node_mask]
                node_teacher = predictions[indices]
                stop = depth >= config.max_depth or len(node_y) < 2 * config.min_labeled_leaf
                if len(node_y):
                    stop = stop or (len(np.unique(node_y)) <= 1 if self.task == "classification"
                                    else np.var(node_y) < 1e-12)
                if not stop:
                    feature_count = max(1, int(np.ceil(
                        config.max_features * len(self.feature_names))))
                    features = tree_rng.choice(self.feature_names, feature_count,
                                               replace=False).tolist()
                    candidates = generate_candidates(node_df, features, numeric_cols,
                                                      config, metadata)
                    candidates = valid_candidates(candidates, node_df, node_mask, config)
                    if candidates:
                        examples = node_df.loc[node_mask].copy()
                        examples[self.target] = raw_y[indices][node_mask]
                        examples = examples.replace([np.inf, -np.inf], np.nan)
                        examples = examples.astype(object).where(pd.notna(examples), None)
                        priors = semantic_priors(
                            llm, candidates, node_df, examples, self.target, self.task,
                            self.labels, path, metadata,
                        )
                        masks = [apply_candidate(node_df, c)[0] for c in candidates]
                        utilities = [
                            split_utilities(self.task, node_y, node_teacher, node_mask,
                                            mask, n_classes) for mask in masks
                        ]
                        posterior = posterior_probabilities(priors, utilities, config)
                        best = int(np.argmax(posterior))
                        candidate, left = candidates[best], masks[best]
                        rule = candidate_rule(candidate)
                        return TreeNode(
                            split=candidate,
                            left=build(indices[left], depth + 1, path + [rule]),
                            right=build(indices[~left], depth + 1, path + [f"NOT ({rule})"]),
                        )
                return TreeNode(prediction=hybrid_leaf(
                    self.task, node_y, node_teacher, config, n_classes))
            self.trees.append(build(np.arange(len(df)), 0, []))
        return self

    def predict(self, X):
        if not self.trees:
            raise RuntimeError("Forest has not been fitted")
        augmented = apply_feature_specs(X, self.feature_specs)[self.feature_names]
        shape = (len(augmented), len(self.labels)) if self.task == "classification" else (len(augmented),)
        result = np.zeros(shape, dtype=float)
        def route(node, indices, output):
            if node.split is None:
                output[indices] = node.prediction
                return
            left, _ = apply_candidate(augmented.iloc[indices], node.split)
            if left.any():
                route(node.left, indices[left], output)
            if (~left).any():
                route(node.right, indices[~left], output)
        for tree in self.trees:
            output = np.empty(shape, dtype=float)
            route(tree, np.arange(len(augmented)), output)
            result += output
        result /= len(self.trees)
        return normalize_probabilities(result) if self.task == "classification" else result

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path):
        with Path(path).open("rb") as handle:
            return pickle.load(handle)
