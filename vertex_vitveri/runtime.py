from __future__ import annotations

import json
import os
import sys
import time

from .config import AdapterConfig


def install_source_paths(config: AdapterConfig):
    paths = (
        config.vitveri_root / "src",
        config.alpha_beta_crown_root,
        config.alpha_beta_crown_root / "auto_LiRPA",
    )
    for path in reversed(paths):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _load_image_context(config: AdapterConfig, task_id: int):
    if not config.weight_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {config.weight_path}")
    install_source_paths(config)

    import torch
    from torchvision import datasets, transforms

    from vitveri.data.evaluation import evaluation_indices
    from vitveri.data.tasks import decode_image_job_index
    from vitveri.models import load_vit
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_vit(config.model, config.weight_path, device)
    transform = transforms.Compose(
        [
            transforms.Resize((config.model["image_size"], config.model["image_size"])),
            transforms.Grayscale(num_output_channels=config.model["channels"]),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.MNIST(
        root=config.data_root,
        train=False,
        download=config.training.get("download_mnist", True),
        transform=transform,
    )
    indices = evaluation_indices(
        config.verification,
        dataset_size=len(dataset),
        config_path=config.config_path,
    )
    image_index = decode_image_job_index(task_id, indices)
    image, true_label = dataset[image_index]
    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))
    predicted_label = int(logits.argmax(dim=1).item())
    return model, image.to(device), int(true_label), predicted_label, image_index, device


def _result_dir(config):
    from vitveri.run_metadata import vertex_softmax_run_dir

    return config.vitveri_root / vertex_softmax_run_dir(
        config.verification,
        config.model,
        config.epsilon,
    )


def _write_metadata(config, output_dir, job_mode):
    from vitveri.run_metadata import build_run_metadata, infer_model_name, write_run_metadata

    metadata = build_run_metadata(
        method="Vertex-Softmax",
        variant=config.method,
        experiment_name=config.experiment_name,
        model_name=infer_model_name(config.model, config.verification),
        epsilon=config.epsilon,
        depth=config.model["depth"],
        config_path=config.config_path,
        output_dir=output_dir,
        verification_config=config.verification,
        job_mode=job_mode,
    )
    metadata["vertex_softmax"] = config.vertex
    metadata["alpha_beta_crown_root"] = str(config.alpha_beta_crown_root)
    return write_run_metadata(output_dir, metadata, config.config_path)


def run_adapter_check(config: AdapterConfig, task_id: int):
    _model, _image, true_label, predicted_label, image_index, _device = _load_image_context(config, task_id)

    output_dir = _result_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_metadata(config, output_dir, "adapter_check")
    result = {
        "status": "adapter_check_complete",
        "method": config.method,
        "image_index": image_index,
        "task_id": task_id,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "epsilon": config.epsilon,
        "weight_path": str(config.weight_path),
        "job_id": os.environ.get("JOB_ID", ""),
    }
    result_path = output_dir / f"adapter_check_task_{task_id}.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["output_path"] = str(result_path)
    return result


def run_certification(config: AdapterConfig, task_id: int):
    if not (config.alpha_beta_crown_root / "auto_LiRPA").is_dir():
        raise FileNotFoundError(f"auto_LiRPA source tree is missing under {config.alpha_beta_crown_root}")
    model, image, true_label, predicted_label, image_index, device = _load_image_context(config, task_id)
    from .certifier import certify_image
    from .crown_provider import run_crown_bounds

    started = time.perf_counter()
    crown_bounds = run_crown_bounds(
        config,
        model,
        image.unsqueeze(0),
        true_label,
        predicted_label,
        image_index,
        device,
    )
    crown_elapsed = time.perf_counter() - started

    certificate = certify_image(
        model,
        image,
        config.epsilon,
        predicted_label,
        config.method,
        crown_method=config.vertex["crown_method"],
        alpha_iters=int(config.vertex["alpha_iters"]),
        crown_bounds=crown_bounds,
    )
    postprocess_elapsed = certificate.pop("elapsed_sec")
    certificate["crown_elapsed_sec"] = crown_elapsed
    certificate["postprocess_elapsed_sec"] = postprocess_elapsed
    certificate["elapsed_sec"] = time.perf_counter() - started
    result = {
        "status": "complete",
        "method": config.method,
        "image_index": image_index,
        "task_id": task_id,
        "true_label": true_label,
        "predicted_label": predicted_label,
        "epsilon": config.epsilon,
        "weight_path": str(config.weight_path),
        "device": str(device),
        "job_id": os.environ.get("JOB_ID", ""),
        "attention_nodes": crown_bounds.attention_layers,
        **certificate,
    }
    output_dir = _result_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_metadata(config, output_dir, "image")
    result_path = output_dir / f"vertex_result_idx{image_index}.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["output_path"] = str(result_path)
    return result
