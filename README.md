# LEAF

Code for **Semantic Priors Meet Statistical Evidence: Robust Forests for Few-Shot Tabular Learning**.
This repository contains the main classification/regression method.

1. Install:

   ```bash
   conda env create -f environment.yml
   conda activate leaf
   ```

2. Prepare the 16 benchmarks with network access:

   ```bash
   python scripts/prepare_data.py --out-dir data
   ```

   Heart, CDC Diabetes, and Insurance require a configured Kaggle account.
   Existing CSVs may be placed at `data/<dataset>.csv`, with the target in the last column.
   Optional column meanings use `data/<dataset>-metadata.json`. Dataset names are in `leaf/config.py`.

3. Set the v2.5 default TabPFN checkpoint paths, then run with the default API model, `gpt-5.4-nano`:

   ```bash
   export TABPFN_CLASSIFIER_CKPT="/path/to/tabpfn-v2.5-classifier-v2.5_default.ckpt"
   export TABPFN_REGRESSOR_CKPT="/path/to/tabpfn-v2.5-regressor-v2.5_default.ckpt"
   export OPENAI_API_KEY="<your-key>"
   # Optional: export OPENAI_BASE_URL="<your-endpoint>"
   bash scripts/run_classification.sh
   bash scripts/run_regression.sh
   ```

   For local Qwen:

   ```bash
   export LLM_BACKEND=local
   export QWEN_MODEL_PATH="/path/to/Qwen3.5-4B"
   bash scripts/run_classification.sh
   bash scripts/run_regression.sh
   ```

The scripts run 4/8/16 total labeled examples and seeds 0–9, with an equal RF/XGBoost/TabPFN teacher average.
The fixed configuration uses 9 trees, depth 4, 8 derived features, MAP selection, and empirical/teacher leaves
(`kappa=2, gamma=1`). Metrics are macro OVR AUC and RMSE divided by test-target standard deviation.

Set `TABPFN_CLASSIFIER_CKPT` and `TABPFN_REGRESSOR_CKPT` to the v2.5 default checkpoints used by the teacher configuration.
Prepare Qwen and TabPFN weights before running on offline compute nodes; submit the shell scripts through your scheduler.

Outputs are under `outputs/<task>/<model>/`, including models, `summary.csv`, and `summary_mean.csv`.
`--resume` reuses completed runs with matching inputs/configuration.
To run a subset, append e.g. `--datasets blood --shots 8 --seeds 0` to the classification script.
Data, weights, caches, and results are excluded from Git.

Formula checks: `python -B -m unittest discover -s tests -v`.
