import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vertex_vitveri.config import load_adapter_config


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


if __name__ == "__main__":
    unittest.main()
