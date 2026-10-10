import argparse
import os
from .config import CLASSIFICATION_DATASETS, REGRESSION_DATASETS, LLMConfig
from .runner import run_pipeline

def main():
    parser = argparse.ArgumentParser(description="Run the LEAF main method.")
    parser.add_argument("--task", choices=["classification", "regression"], required=True)
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--shots", type=int, nargs="+", choices=[4, 8, 16], default=[4, 8, 16])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--llm-backend", choices=["api", "local"], default="api")
    parser.add_argument("--llm-model")
    parser.add_argument("--llm-cache-dir", default="outputs/llm_cache")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--llm-device", default="auto")
    parser.add_argument("--teacher-device", default="auto")
    parser.add_argument("--tabpfn-checkpoint")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or any(seed < 0 for seed in args.seeds):
        parser.error("max-new-tokens must be positive and seeds must be nonnegative")
    model = args.llm_model or (
        "gpt-5.4-nano" if args.llm_backend == "api" else
        os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen3.5-4B")
    )
    datasets = args.datasets or list(CLASSIFICATION_DATASETS if args.task == "classification"
                                    else REGRESSION_DATASETS)
    checkpoint = args.tabpfn_checkpoint or os.environ.get(
        "TABPFN_CLASSIFIER_CKPT" if args.task == "classification" else "TABPFN_REGRESSOR_CKPT")
    if not checkpoint:
        parser.error("Set TABPFN_CLASSIFIER_CKPT / TABPFN_REGRESSOR_CKPT or --tabpfn-checkpoint")
    config = LLMConfig(model=model, backend=args.llm_backend,
                       max_new_tokens=args.max_new_tokens, cache_dir=args.llm_cache_dir,
                       device=args.llm_device)
    run_pipeline(args.data_root, datasets, args.task, args.shots, args.seeds,
                 args.output_root, config, args.teacher_device, checkpoint, args.resume)

if __name__ == "__main__":
    main()
