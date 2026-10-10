#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python -m leaf.cli \
  --task classification \
  --datasets adult blood breast-w credit-g cdc_diabetes heart gallstone \
  --data-root "${DATA_ROOT:-data}" \
  --output-root "${OUTPUT_ROOT:-outputs}" \
  --shots 4 8 16 --seeds 0 1 2 3 4 5 6 7 8 9 \
  --llm-backend "${LLM_BACKEND:-api}" \
  --resume "$@"
