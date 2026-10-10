"""LLM-proposed deterministic feature mapping, shared by both tasks."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import numpy as np
import pandas as pd
from .utils import json_default
import json

_ALLOWED_OPS = {"ratio", "difference", "sum", "product", "log1p_abs", "missing"}

@dataclass(frozen=True)
class DerivedFeatureSpec:
    name: str
    op: str
    cols: list[str]

def feature_prompt(columns, numeric_cols, metadata, target, task, budget):
    # Metadata-only proposals are shared across shots/seeds. No hidden labels or
    # held-out feature statistics enter this global proposal prompt.
    schema = [{"token": f"c{i}", "name": col,
               "type": "numeric" if col in numeric_cols else "categorical",
               "meaning": metadata.get(col, "")} for i, col in enumerate(columns)]
    context = metadata.get("__dataset_description__", "")
    target_context = metadata.get("__target_description__", target)
    return (
        "You are an expert designing deterministic derived features for few-shot tabular learning.\n"
        f"Target column: {target}. Task type: {task}. Task description: {context} {target_context}.\n"
        f"Columns: {json.dumps(schema, default=json_default)}\n"
        "Use feature meanings when available. Prefer semantically meaningful cross-column comparisons, "
        "interactions, consistency indicators, and relational abstractions over arbitrary algebra.\n"
        "Each derived feature must be computable deterministically from the original columns. "
        "Do not use the target or labels. Return a JSON list only, with name, op, cols per item.\n"
        "Allowed operations: ratio, difference, sum, product, log1p_abs, missing. "
        "Only missing may use categorical columns. Unary operations require one column; others require "
        "at least two numeric columns. ratio is c0/(c1+c2+...+eps); difference is c0-c1-c2-...; "
        "sum and product combine all listed columns.\n"
        f"Return at most {budget} items."
    )

def parse_feature_specs(raw, columns, numeric_cols, budget):
    if not isinstance(raw, list):
        return []
    mapping = {f"c{i}": col for i, col in enumerate(columns)}
    used = set(columns)
    specs = []
    for item in raw:
        if not isinstance(item, dict) or item.get("op") not in _ALLOWED_OPS:
            continue
        tokens = item.get("cols")
        if not isinstance(tokens, list):
            continue
        cols = [mapping.get(str(c), str(c)) for c in tokens]
        op = item["op"]
        if not cols or any(c not in columns for c in cols):
            continue
        if (op in {"log1p_abs", "missing"} and len(cols) != 1) or (
            op not in {"log1p_abs", "missing"} and len(cols) < 2
        ):
            continue
        if op != "missing" and any(c not in numeric_cols for c in cols):
            continue
        name = str(item.get("name", "")).strip()
        if not name or name.startswith("__") or name in used:
            continue
        used.add(name)
        specs.append(DerivedFeatureSpec(name, op, cols))
        if len(specs) >= budget:
            break
    return specs

def propose_features(llm, columns, numeric_cols, metadata, target, task, budget=8):
    if budget <= 0:
        return []
    prompt = feature_prompt(columns, numeric_cols, metadata, target, task, budget)
    return parse_feature_specs(llm.generate_json(prompt, default=[]),
                               columns, numeric_cols, budget)

def apply_feature_specs(df, specs):
    out = df.copy()
    for spec in specs:
        if spec.op == "missing":
            value = df[spec.cols[0]].isna().astype(float)
        else:
            values = [pd.to_numeric(df[c], errors="coerce").replace(
                [np.inf, -np.inf], np.nan) for c in spec.cols]
            value = values[0].copy()
            if spec.op == "log1p_abs":
                value = np.log1p(np.abs(value))
            elif spec.op == "ratio":
                denominator = values[1].copy()
                for other in values[2:]:
                    denominator = denominator + other
                value = value / (denominator.replace(0, np.nan) + 1e-6)
            else:
                for other in values[1:]:
                    if spec.op == "difference":
                        value = value - other
                    elif spec.op == "sum":
                        value = value + other
                    elif spec.op == "product":
                        value = value * other
                    else:
                        raise ValueError(f"Unsupported feature operation: {spec.op}")
        out[spec.name] = value.replace([np.inf, -np.inf], np.nan)
    return out

def specs_as_dicts(specs):
    return [asdict(spec) for spec in specs]
