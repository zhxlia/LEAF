"""Prepare teacher signals, induce LEAF, and evaluate benchmark runs."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
from .config import LEAFConfig
from .data import feature_types, prepare_dataset, read_dataset, validate_datasets
from .features import propose_features, specs_as_dicts
from .forest import LEAFForest
from .llm import build_llm
from .metrics import evaluate
from .teachers import TeacherEnsemble
from .utils import fingerprint, load_json, save_json

def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def run_pipeline(data_root, datasets, task, shots, seeds, output_root, llm_config,
                 teacher_device="auto", tabpfn_checkpoint=None, resume=False):
    validate_datasets(task, datasets)
    config = LEAFConfig()
    llm = build_llm(llm_config)
    model_key = Path(llm_config.model.rstrip("/")).name
    root = Path(output_root) / task / model_key
    source_hash = fingerprint({
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob("*.py"))
    })
    checkpoint_hash = None
    if tabpfn_checkpoint:
        checkpoint_hash = file_hash(tabpfn_checkpoint)
    results = []
    for dataset in datasets:
        df, metadata = read_dataset(data_root, dataset)
        target = df.columns[-1]
        raw_columns = df.columns[:-1].tolist()
        numeric, _ = feature_types(df[raw_columns])
        data_hash = hashlib.sha256(pd.util.hash_pandas_object(
            df, index=True).values.tobytes()).hexdigest()
        # A metadata-only proposal is reused across shots and seeds.
        specs = propose_features(llm, raw_columns, numeric, metadata,
                                 target, task, config.max_derived_features)
        for shot in shots:
            for seed in seeds:
                run_dir = root / dataset / f"shot_{shot}" / f"seed_{seed}"
                manifest_path = run_dir / "manifest.json"
                signature = fingerprint({
                    "source": source_hash, "data": data_hash, "columns": df.columns.tolist(),
                    "metadata": metadata, "config": asdict(config), "llm": asdict(llm_config),
                    "teacher_device": teacher_device, "tabpfn_checkpoint": checkpoint_hash,
                    "api_endpoint": os.environ.get("OPENAI_BASE_URL", os.environ.get("BASE_URL", "")),
                    "dataset": dataset, "task": task, "shot": shot, "seed": seed,
                    "feature_specs": specs_as_dicts(specs),
                })
                artifacts = [run_dir / "model.pkl", run_dir / "metrics.json",
                             run_dir / "predictions.npz"]
                if resume and manifest_path.exists() and all(p.is_file() for p in artifacts):
                    manifest = load_json(manifest_path)
                    if manifest.get("signature") == signature and manifest.get("complete"):
                        metrics = load_json(run_dir / "metrics.json")
                        results.append({"dataset": dataset, "shot": shot, "seed": seed, **metrics})
                        print(f"{dataset} shot={shot} seed={seed}: cached {metrics}", flush=True)
                        continue
                random.seed(seed)
                np.random.seed(seed)
                run_dir.mkdir(parents=True, exist_ok=True)
                save_json(manifest_path, {"signature": signature, "complete": False})
                prepared = prepare_dataset(df, metadata, task, shot, seed)
                teacher = TeacherEnsemble(task, seed, labels=prepared.labels,
                                          tabpfn_checkpoint=tabpfn_checkpoint,
                                          device=teacher_device)
                teacher.fit(prepared.labeled[raw_columns], prepared.labeled[target])
                train_all = pd.concat(
                    [prepared.labeled[raw_columns], prepared.unlabeled], ignore_index=True)
                teacher_predictions = teacher.predict(train_all)
                forest = LEAFForest(task, target, prepared.labels, config, seed)
                forest.fit(prepared.labeled, prepared.unlabeled, teacher_predictions,
                           llm, specs, metadata)
                forest.save(run_dir / "model.pkl")
                prediction = forest.predict(prepared.test[raw_columns])
                metrics = evaluate(task, prepared.test[target], prediction, prepared.labels)
                save_json(run_dir / "metrics.json", metrics)
                save_json(run_dir / "feature_specs.json", specs_as_dicts(specs))
                np.savez_compressed(run_dir / "predictions.npz",
                                    y_true=prepared.test[target].to_numpy(),
                                    prediction=prediction)
                save_json(manifest_path, {
                    "signature": signature, "complete": True,
                    "dataset": dataset, "task": task, "shot": shot, "seed": seed,
                    "config": asdict(config), "llm": asdict(llm_config),
                    "split_sizes": {"labeled": len(prepared.labeled),
                                    "unlabeled": len(prepared.unlabeled),
                                    "test": len(prepared.test)},
                })
                results.append({"dataset": dataset, "shot": shot, "seed": seed, **metrics})
                print(f"{dataset} shot={shot} seed={seed}: {metrics}", flush=True)
    result_frame = pd.DataFrame(results)
    root.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(root / "summary.csv", index=False)
    value = "auc" if task == "classification" else "nrmse"
    aggregate = result_frame.groupby(["dataset", "shot"])[value].agg(["mean", "std", "count"])
    aggregate.to_csv(root / "summary_mean.csv")
    return results
