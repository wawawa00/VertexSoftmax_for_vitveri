from __future__ import annotations

import json

from .cli import parse_args
from .config import load_adapter_config
from .direct_crown_provider import run_direct_crown_bounds
from .runtime import run_adapter_check


def _run_direct_certification(config, task_id):
    from pathlib import Path

    from . import certifier, crown_provider, runtime

    original_provider = crown_provider.run_crown_bounds
    original_query_inverse = certifier._attention_input_bounds_from_query
    original_norm_inverse = certifier._centered_block_input_bounds
    original_result_dir = runtime._result_dir
    original_source = config.vertex.get("intermediate_bound_source")

    def identity_query(_model, lower, upper):
        return lower, upper, None

    def identity_norm(_model, lower, upper):
        return lower, upper

    def direct_result_dir(adapter_config):
        return original_result_dir(adapter_config) / "direct_init_crown"

    crown_provider.run_crown_bounds = run_direct_crown_bounds
    certifier._attention_input_bounds_from_query = identity_query
    certifier._centered_block_input_bounds = identity_norm
    runtime._result_dir = direct_result_dir
    config.vertex["intermediate_bound_source"] = "direct_init_crown"
    try:
        result = runtime.run_certification(config, task_id)
        result["intermediate_bound_source"] = "direct_init_crown"
        Path(result["output_path"]).write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result
    finally:
        crown_provider.run_crown_bounds = original_provider
        certifier._attention_input_bounds_from_query = original_query_inverse
        certifier._centered_block_input_bounds = original_norm_inverse
        runtime._result_dir = original_result_dir
        if original_source is None:
            config.vertex.pop("intermediate_bound_source", None)
        else:
            config.vertex["intermediate_bound_source"] = original_source


def main(argv=None):
    args = parse_args(argv)
    config = load_adapter_config(args.config)
    if args.preflight:
        result = config.as_dict()
    elif args.check_adapter:
        result = run_adapter_check(config, args.task_id)
    else:
        result = _run_direct_certification(config, args.task_id)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
