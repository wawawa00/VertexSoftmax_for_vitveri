import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vertex_vitveri.config import load_adapter_config
from vertex_vitveri.crown_provider import discover_attention_layers


class BoundMatMul:
    def __init__(self, name, inputs, *, nonlinear=True):
        self.name = name
        self.inputs = inputs
        self.perturbed = nonlinear
        self.requires_input_bounds = [0] if nonlinear else []


class BoundExp(BoundMatMul):
    pass


class BoundRelu(BoundMatMul):
    pass


class BoundReduceMean(BoundMatMul):
    pass


class BoundSub(BoundMatMul):
    pass


class FakeNode:
    def __init__(self, name, inputs=()):
        self.name = name
        self.inputs = list(inputs)
        self.perturbed = False
        self.requires_input_bounds = []


class FakeNet:
    def __init__(self, nodes):
        self._nodes = nodes

    def nodes(self):
        return self._nodes


class VitveriAdapterConfigTest(unittest.TestCase):
    def test_loads_all_adapter_inputs_from_one_central_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            vitveri = workspace / "vitveri"
            (vitveri / "src" / "vitveri").mkdir(parents=True)
            (vitveri / "src" / "vitveri" / "__init__.py").write_text("", encoding="utf-8")
            crown = workspace / "GenBaB_for_vitveri" / "alpha-beta-CROWN"
            (crown / "auto_LiRPA").mkdir(parents=True)
            weights = vitveri / "weights" / "model.pth"
            weights.parent.mkdir()
            weights.write_bytes(b"checkpoint")
            config_path = vitveri / "configs" / "cone" / "small.yaml"
            config_path.parent.mkdir(parents=True)
            config_path.write_text("placeholder", encoding="utf-8")
            parsed = {
                "name": "small_vertex",
                "model": {"layer_norm_type": "no_var", "pool": "cls", "depth": 1},
                "training": {"data_root": "data"},
                "verification": {
                    "epsilon": 0.02,
                    "seed": 42,
                    "num_samples": 50,
                    "weight_path": "weights/model.pth",
                    "vertex_softmax": {"method": "crown_objective_vertex_hybrid", "alpha_iters": 12},
                },
            }

            with patch("vertex_vitveri.config._load_central_config", return_value=parsed):
                with patch.dict(os.environ, {"VITVERI_ROOT": str(vitveri)}, clear=False):
                    config = load_adapter_config(config_path)

        self.assertEqual(config.vitveri_root, vitveri.resolve())
        self.assertEqual(config.alpha_beta_crown_root, crown.resolve())
        self.assertEqual(config.weight_path, weights.resolve())
        self.assertEqual(config.method, "crown_objective_vertex_hybrid")
        self.assertEqual(config.vertex["alpha_iters"], 12)
        self.assertEqual(config.sample_size, 50)

    def test_rejects_architecture_not_supported_by_adapter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "src" / "vitveri").mkdir(parents=True)
            config_path = root / "config.yaml"
            config_path.write_text("placeholder", encoding="utf-8")
            parsed = {
                "model": {"layer_norm_type": "standard", "pool": "cls"},
                "training": {},
                "verification": {"epsilon": 0.01, "weight_path": "missing.pth"},
            }
            with patch("vertex_vitveri.config._load_central_config", return_value=parsed):
                with self.assertRaisesRegex(ValueError, "layer_norm_type=no_var"):
                    load_adapter_config(config_path, vitveri_root=root)


class CrownProviderTest(unittest.TestCase):
    def test_discovers_scaled_attention_nodes(self):
        block_input = FakeNode("/block_input")
        mean = BoundReduceMean("/mean", [block_input], nonlinear=False)
        centered = BoundSub("/centered", [block_input, mean], nonlinear=False)
        q = FakeNode("/q", [centered])
        k = FakeNode("/k", [centered])
        score = BoundMatMul("/qk", [q, k])
        scaled = FakeNode("/scaled")
        reduce_max = FakeNode("/max")
        shifted = FakeNode("/shifted", [scaled, reduce_max])
        exp = BoundExp("/exp", [shifted])
        probabilities = FakeNode("/softmax/mul")
        value = FakeNode("/v")
        context = BoundMatMul("/context", [probabilities, value])
        relu_input = FakeNode("/relu_input")
        relu = BoundRelu("/relu", [relu_input])

        layers = discover_attention_layers(
            FakeNet(
                [
                    block_input,
                    mean,
                    centered,
                    q,
                    k,
                    score,
                    scaled,
                    reduce_max,
                    shifted,
                    exp,
                    probabilities,
                    value,
                    context,
                    relu,
                ]
            ),
            expected_depth=1,
        )

        self.assertEqual(
            layers[0],
            {
                "q": "/q",
                "k": "/k",
                "score_scaled": "/scaled",
                "score_scaled_scale": "none",
                "score_fallback_final_node": "/qk",
                "score_fallback_scale": "by_dim",
                "v": "/v",
                "softmax": "/softmax/mul",
                "relu_input": "/relu_input",
                "block_input": "/block_input",
            },
        )

    def test_rejects_incomplete_attention_layer(self):
        q = FakeNode("/q")
        k = FakeNode("/k")
        score = BoundMatMul("/qk", [q, k])

        with self.assertRaisesRegex(ValueError, "missing nodes"):
            discover_attention_layers(FakeNet([q, k, score]), expected_depth=1)


if __name__ == "__main__":
    unittest.main()
