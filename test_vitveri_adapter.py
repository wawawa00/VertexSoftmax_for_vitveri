import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

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


class BoundTranspose(BoundMatMul):
    pass


class FakeNode:
    def __init__(self, name, inputs=(), *, perturbed=False):
        self.name = name
        self.inputs = list(inputs)
        self.perturbed = perturbed
        self.requires_input_bounds = []


class FakeNet:
    def __init__(self, nodes):
        self._nodes = nodes

    def nodes(self):
        return self._nodes


class VitveriAdapterConfigTest(unittest.TestCase):
    def test_explicit_output_dir_takes_precedence_without_editing_config(self):
        from vertex_vitveri.cli import parse_args
        from vertex_vitveri.runtime import _result_dir
        args = parse_args(["--config", "config.yaml", "--task-id", "1", "--certify", "--output-dir", "result"])
        self.assertEqual(args.output_dir, "result")
        directory = Path("/explicit/result")
        self.assertEqual(_result_dir(SimpleNamespace(output_dir=directory)), directory)

    def test_unified_run_refuses_existing_result_before_crown(self):
        from vertex_vitveri import runtime
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "auto_LiRPA").mkdir()
            config = SimpleNamespace(alpha_beta_crown_root=root, crown={"general": {}})
            for result in (root / "image_1824/result.json", root / "vertex_result_idx1824.json"):
                with self.subTest(path=result):
                    result.parent.mkdir(parents=True, exist_ok=True)
                    result.write_text("previous result")
                    with patch.object(runtime, "_load_image_context", return_value=(None, None, 2, 2, 1824, "cpu")), \
                         patch.object(runtime, "_result_dir", return_value=root), \
                         patch("vertex_vitveri.crown_provider.run_crown_bounds") as bounds:
                        with self.assertRaises(FileExistsError):
                            runtime.run_certification(config, 1)
                    bounds.assert_not_called()
                    self.assertEqual(result.read_text(), "previous result")
                    result.unlink()

    def test_certification_saves_result_beside_each_images_crown_inputs(self):
        from vertex_vitveri import runtime
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "auto_LiRPA").mkdir()
            config = SimpleNamespace(
                alpha_beta_crown_root=root, epsilon=0.019, weight_path=root / "weights.pth",
                method="crown_objective_vertex_hybrid", vertex={"crown_method": "CROWN", "alpha_iters": 20},
                crown_artifacts=[{"path": "property.vnnlib", "sha256": "test"}],
            )
            image = Mock()
            bound_data = SimpleNamespace(attention_layers=[{"q": "/q"}])
            certificate = {"certified": True, "target_lowers": [1.0, 2.0], "elapsed_sec": 0.25}
            for task, image_index in ((1, 1824), (2, 3109)):
                with patch.object(runtime, "_load_image_context", return_value=(None, image, 2, 2, image_index, "cpu")), \
                     patch.object(runtime, "_result_dir", return_value=root), \
                     patch("vertex_vitveri.crown_provider.run_crown_bounds", return_value=bound_data), \
                     patch("vertex_vitveri.certifier.certify_image", return_value=dict(certificate)), \
                     patch.object(runtime, "_write_metadata") as metadata:
                    result = runtime.run_certification(config, task)
                path = root / f"image_{image_index}/result.json"
                saved = json.loads(path.read_text())
                self.assertEqual(result["output_path"], str(path))
                self.assertEqual(saved["image_index"], image_index)
                self.assertEqual(saved["task_id"], task)
                self.assertEqual(saved["certified"], certificate["certified"])
                self.assertEqual(saved["target_lowers"], certificate["target_lowers"])
                self.assertEqual(saved["postprocess_elapsed_sec"], 0.25)
                self.assertEqual(saved["attention_nodes"], bound_data.attention_layers)
                self.assertEqual(saved["crown_artifacts"], config.crown_artifacts)
                self.assertFalse((root / f"vertex_result_idx{image_index}.json").exists())
                self.assertEqual(metadata.call_count, 2)
                metadata.assert_called_with(config, root, "image")

    def test_incompatible_metadata_stops_before_crown(self):
        from vertex_vitveri import runtime
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "auto_LiRPA").mkdir()
            config = SimpleNamespace(alpha_beta_crown_root=root)
            with patch.object(runtime, "_load_image_context", return_value=(None, Mock(), 2, 2, 1824, "cpu")), \
                 patch.object(runtime, "_result_dir", return_value=root), \
                 patch.object(runtime, "_write_metadata", side_effect=ValueError("Run input differs")), \
                 patch("vertex_vitveri.crown_provider.run_crown_bounds") as bounds:
                with self.assertRaisesRegex(ValueError, "Run input differs"):
                    runtime.run_certification(config, 1)
            bounds.assert_not_called()
            self.assertFalse((root / "image_1824").exists())

    def test_unified_yaml_and_inputs_are_retained_by_both_providers(self):
        import yaml
        from vertex_vitveri import crown_provider, direct_crown_provider

        class ReachedVerifier(Exception):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "weights.pth").write_bytes(b"weights")
            config = SimpleNamespace(
                config_path=root / "experiment.yaml", epsilon=0.02,
                verification={"abcrown_entry": "abcrown.py", "weight_path": "weights.pth"},
                model={"image_size": 8, "patch_size": 4, "num_classes": 10, "channels": 1,
                       "dim": 8, "depth": 1, "heads": 1, "mlp_dim": 16,
                       "layer_norm_type": "no_var", "pool": "cls", "dim_head": 8},
                crown={"general": {}, "model": {}, "solver": {"batch_size": 50}},
                cone={"crown_overrides": {"general": {"complete_verifier": "skip"}}},
                crown_artifacts=[], resolve_external_path=lambda value: root / value,
            )
            for provider in (crown_provider, direct_crown_provider):
                output = root / provider.__name__.rsplit(".", 1)[-1]
                verifier = Mock(side_effect=ReachedVerifier)
                function = provider.run_crown_bounds if provider is crown_provider else provider.run_direct_crown_bounds
                with patch.object(provider, "load_abcrown_class", return_value=verifier), \
                     patch.object(crown_provider, "generate_vnnlib", return_value="fixture"), \
                     patch("vertex_vitveri.runtime._result_dir", return_value=output), \
                     self.assertRaises(ReachedVerifier):
                    function(config, None, None, 2, 2, 1824, "cpu")
                path = Path(verifier.call_args.kwargs["args"][1])
                self.assertEqual(path, output / "image_1824/crown/abcrown.yaml")
                engine = yaml.safe_load(path.read_text())
                self.assertEqual(engine["solver"]["batch_size"], 50)
                self.assertEqual(engine["general"]["complete_verifier"], "skip")
                self.assertTrue(Path(engine["general"]["csv_name"]).is_file())
                self.assertTrue((path.parent / "property.vnnlib").is_file())
                before = {file.name: file.read_bytes() for file in path.parent.iterdir()}
                with patch.object(provider, "load_abcrown_class", return_value=verifier), \
                     patch("vertex_vitveri.runtime._result_dir", return_value=output), \
                     self.assertRaises(FileExistsError):
                    function(config, None, None, 2, 2, 1824, "cpu")
                verifier.assert_called_once()
                self.assertEqual(before, {file.name: file.read_bytes() for file in path.parent.iterdir()})
            self.assertEqual(len(config.crown_artifacts), 6)

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
            manifest = vitveri / "indices.txt"
            manifest.write_text("7\n2\n", encoding="utf-8")
            parsed = {
                "name": "small_vertex",
                "crown": {},
                "model": {"layer_norm_type": "no_var", "pool": "cls", "depth": 1},
                "training": {},
                "verification": {
                    "epsilon": 0.02,
                    "data_root": "../../data",
                    "index_file": str(manifest),
                    "weight_path": "../../weights/model.pth",
                },
                "vertex_softmax": {"method": "crown_objective_vertex_hybrid", "alpha_iters": 12},
            }

            with patch("vertex_vitveri.config._load_central_config", return_value=parsed):
                with patch.dict(os.environ, {"VITVERI_ROOT": str(vitveri)}, clear=False):
                    config = load_adapter_config(config_path)
                    sample_size = config.sample_size

        self.assertEqual(config.vitveri_root, vitveri.resolve())
        self.assertEqual(config.alpha_beta_crown_root, crown.resolve())
        self.assertEqual(config.weight_path, weights.resolve())
        self.assertEqual(config.data_root, (vitveri / "data").resolve())
        self.assertEqual(config.method, "crown_objective_vertex_hybrid")
        self.assertEqual(config.vertex["alpha_iters"], 12)
        self.assertEqual(sample_size, 2)

    def test_rejects_architecture_not_supported_by_adapter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "src" / "vitveri").mkdir(parents=True)
            config_path = root / "config.yaml"
            config_path.write_text("placeholder", encoding="utf-8")
            parsed = {
                "model": {"layer_norm_type": "standard", "pool": "cls"},
                "crown": {},
                "training": {},
                "verification": {"epsilon": 0.01, "weight_path": "missing.pth"},
            }
            with patch("vertex_vitveri.config._load_central_config", return_value=parsed):
                with self.assertRaisesRegex(ValueError, "layer_norm_type=no_var"):
                    load_adapter_config(config_path, vitveri_root=root)


class CrownProviderTest(unittest.TestCase):
    def test_unified_score_bounds_do_not_recompute_without_perturbation(self):
        from vertex_vitveri.crown_provider import _score_bounds
        net = SimpleNamespace(compute_bounds=Mock())
        with self.assertRaisesRegex(RuntimeError, "Saved score bounds"):
            _score_bounds({"lower_bounds": {}, "upper_bounds": {}}, net, None, None, {})
        net.compute_bounds.assert_not_called()

    def test_discovers_scaled_attention_nodes(self):
        block_input = FakeNode("/block_input")
        mean = BoundReduceMean("/mean", [block_input], nonlinear=False)
        centered = BoundSub("/centered", [block_input, mean], nonlinear=False)
        normalized = FakeNode("/normalized", [centered], perturbed=True)
        q_weight = FakeNode("/q_weight")
        q_projection = BoundMatMul("/q_projection", [normalized, q_weight], nonlinear=False)
        q_projection.perturbed = True
        q = BoundTranspose("/q", [q_projection], nonlinear=False)
        q.perturbed = True
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
                    normalized,
                    q_weight,
                    q_projection,
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
                "score_unscaled": "/qk",
                "score_unscaled_scale": "by_dim",
                "score_fallback_final_node": "/qk",
                "score_fallback_scale": "by_dim",
                "v": "/v",
                "softmax": "/softmax/mul",
                "relu_input": "/relu_input",
                "block_input": "/block_input",
                "attention_input": "/normalized",
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
