from __future__ import annotations

import argparse
import json

from .config import load_adapter_config
from .runtime import run_adapter_check, run_certification


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run Vertex-Softmax from a central vitveri experiment config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--check-adapter", action="store_true")
    parser.add_argument("--certify", action="store_true")
    args = parser.parse_args(argv)
    if sum((args.preflight, args.check_adapter, args.certify)) != 1:
        parser.error("choose exactly one of --preflight, --check-adapter, and --certify")
    if not args.preflight and args.task_id is None:
        parser.error("--task-id is required with --check-adapter or --certify")
    return args


def main(argv=None):
    args = parse_args(argv)
    config = load_adapter_config(args.config)
    if args.preflight:
        result = config.as_dict()
    elif args.check_adapter:
        result = run_adapter_check(config, args.task_id)
    else:
        result = run_certification(config, args.task_id)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
