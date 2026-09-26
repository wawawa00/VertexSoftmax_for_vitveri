from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_METHODS = {
    "CROWN",
    "objective_vertex_crown",
    "crown_objective_vertex_hybrid",
}


def _first_existing(candidates, description):
    checked = []
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate).expanduser().resolve()
        checked.append(path)
        if path.exists():
            return path
    rendered = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(f"Could not locate {description}. Checked:\n{rendered}")


def _find_vitveri_root(config_path, explicit_root=None):
    env_root = os.environ.get("VITVERI_ROOT")
    candidates = [explicit_root, env_root]
    candidates.extend(parent for parent in config_path.parents if (parent / "src" / "vitveri").is_dir())
    candidates.append(Path(__file__).resolve().parents[2] / "vitveri")
    root = _first_existing(candidates, "vitveri repository")
    if not (root / "src" / "vitveri").is_dir():
        raise FileNotFoundError(f"Not a vitveri repository: {root}")
    return root


def _load_central_config(config_path, vitveri_root):
    source_root = vitveri_root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from vitveri.config import load_experiment_config

    return load_experiment_config(config_path)


def _resolve_repository_path(value, *, config_path, vitveri_root, required=True):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = (Path.cwd() / path, vitveri_root / path, config_path.parent / path)
    if required:
        return _first_existing(candidates, str(value))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (vitveri_root / path).resolve()


def _find_alpha_beta_crown(config_path, vitveri_root, verification, vertex):
    explicit = vertex.get("alpha_beta_crown_root") or os.environ.get("ALPHA_BETA_CROWN_ROOT")
    candidates = []
    if explicit:
        explicit_path = Path(explicit).expanduser()
        candidates.extend(
            [
                explicit_path if explicit_path.is_absolute() else vitveri_root / explicit_path,
                config_path.parent / explicit_path,
            ]
        )

    abcrown_entry = verification.get("abcrown_entry")
    if abcrown_entry:
        try:
            entry = _resolve_repository_path(abcrown_entry, config_path=config_path, vitveri_root=vitveri_root)
            candidates.append(entry.parents[1])
        except (FileNotFoundError, IndexError):
            pass
    candidates.append(vitveri_root.parent / "GenBaB_for_vitveri" / "alpha-beta-CROWN")
    for candidate in candidates:
        if candidate is not None and (Path(candidate).expanduser() / "auto_LiRPA").is_dir():
            return Path(candidate).expanduser().resolve()
    return (vitveri_root.parent / "GenBaB_for_vitveri" / "alpha-beta-CROWN").resolve()


@dataclass(frozen=True)
class AdapterConfig:
    config_path: Path
    vitveri_root: Path
    alpha_beta_crown_root: Path
    weight_path: Path
    data_root: Path
    experiment_name: str
    model: dict
    training: dict
    verification: dict
    vertex: dict

    @property
    def method(self):
        return self.vertex["method"]

    @property
    def epsilon(self):
        return float(self.verification["epsilon"])

    @property
    def sample_size(self):
        return int(self.verification.get("num_samples", 50))

    @property
    def seed(self):
        return int(self.verification.get("seed", 42))

    def as_dict(self):
        return {
            "config_path": str(self.config_path),
            "vitveri_root": str(self.vitveri_root),
            "alpha_beta_crown_root": str(self.alpha_beta_crown_root),
            "auto_lirpa_exists": (self.alpha_beta_crown_root / "auto_LiRPA").is_dir(),
            "weight_path": str(self.weight_path),
            "weight_exists": self.weight_path.is_file(),
            "data_root": str(self.data_root),
            "experiment_name": self.experiment_name,
            "method": self.method,
            "epsilon": self.epsilon,
            "sample_size": self.sample_size,
            "seed": self.seed,
            "model": self.model,
            "vertex": self.vertex,
        }


def load_adapter_config(path, *, vitveri_root=None):
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment config does not exist: {config_path}")

    resolved_vitveri_root = _find_vitveri_root(config_path, vitveri_root)
    config = _load_central_config(config_path, resolved_vitveri_root)
    model = config.get("model")
    training = config.get("training")
    verification = config.get("verification")
    if not isinstance(model, dict) or not isinstance(training, dict) or not isinstance(verification, dict):
        raise ValueError("Config requires model, training, and verification mappings")

    vertex = dict(verification.get("vertex_softmax") or {})
    vertex.setdefault("method", "crown_objective_vertex_hybrid")
    vertex.setdefault("crown_method", "CROWN")
    vertex.setdefault("alpha_iters", 20)
    if vertex["method"] not in SUPPORTED_METHODS:
        choices = ", ".join(sorted(SUPPORTED_METHODS))
        raise ValueError(f"Unsupported verification.vertex_softmax.method={vertex['method']!r}; choose one of {choices}")
    if vertex["crown_method"] not in {"CROWN", "alpha-CROWN"}:
        raise ValueError("verification.vertex_softmax.crown_method must be CROWN or alpha-CROWN")
    if model.get("layer_norm_type") != "no_var":
        raise ValueError("The vitveri Vertex adapter currently requires model.layer_norm_type=no_var")
    if model.get("pool") != "cls":
        raise ValueError("The vitveri Vertex adapter currently requires model.pool=cls")

    weight_value = verification.get("weight_path") or training.get("output_path")
    if not weight_value:
        raise ValueError("Config requires verification.weight_path or training.output_path")
    weight_path = _resolve_repository_path(
        weight_value,
        config_path=config_path,
        vitveri_root=resolved_vitveri_root,
        required=False,
    )
    data_value = training.get("data_root", "data")
    data_path = Path(data_value).expanduser()
    if not data_path.is_absolute():
        data_path = resolved_vitveri_root / data_path

    return AdapterConfig(
        config_path=config_path,
        vitveri_root=resolved_vitveri_root,
        alpha_beta_crown_root=_find_alpha_beta_crown(
            config_path,
            resolved_vitveri_root,
            verification,
            vertex,
        ),
        weight_path=weight_path,
        data_root=data_path.resolve(),
        experiment_name=str(config.get("name") or config_path.stem),
        model=dict(model),
        training=dict(training),
        verification=dict(verification),
        vertex=vertex,
    )
