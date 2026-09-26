# Vertex-Softmax

This repository contains the research code for the Vertex-Softmax paper.
It intentionally excludes raw datasets, generated results, private cluster
submission wrappers, logs, notebooks, review notes, and local machine metadata.
It is a script-based source release rather than an installable Python package.

## Contents

- `interval_vertex_softmax.py`: conservative decimal-interval reference evaluator.
- `attention_cert_method_sweep.py`: scalable PyTorch score-box solvers and toy comparisons.
- `benchmark_scalable_vertex_softmax.py`: runtime benchmark for the threshold solver.
- `paper_scalable_sweep.py`: paper-scale toy attention sweep driver.
- `tiny_vit_block_benchmark.py`: small patch-attention and attention-residual benchmark driver.
- `slack_decomposition_ablation.py`: decomposition experiments for bound slack.
- `generate_paper_figures.py`, `make_runtime_breakdown_table.py`: figure/table generation helpers.
- `full_mlp_vertex_validation_grid.py`: local validation grid wrapper for full-block experiments.
- `test_interval_vertex_softmax.py`, `test_scalable_vertex_softmax.py`: regression checks.

## Quick Check

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python test_interval_vertex_softmax.py
python test_scalable_vertex_softmax.py
```

## vitveri Adapter

The fork accepts the central experiment YAML owned by the sibling `vitveri`
repository. The adapter resolves the existing checkpoint, MNIST sampling seed,
epsilon, and the sibling alpha-beta-CROWN source tree from that one file.

```bash
PYTHONPATH=. python -m vertex_vitveri.cli \
  --config ../vitveri/configs/cone/small_depth1_eps002_cone.yaml \
  --preflight
```

The first integration stage performs an adapter check for one image: it loads
the shared checkpoint and records the selected image and clean prediction. The
certificate path then applies the upstream exact score-box primitive to the
final attention block, with an ABCROWN boundary provider owned by this
repository and an affine/ReLU suffix for the CLS-token classifier. The central
vitveri YAML and checkpoint are shared inputs; the CROWN extraction code is not
imported from vitveri at run time. The current end-to-end certificate supports
the depth-1 model while the adapter is being extended to deeper prefixes.

```bash
PYTHONPATH=. python -m vertex_vitveri.cli \
  --config ../vitveri/configs/cone/small_depth1_eps002_cone.yaml \
  --task-id 1 \
  --check-adapter
```

```bash
PYTHONPATH=. python -m vertex_vitveri.cli \
  --config ../vitveri/configs/cone/small_depth1_eps002_cone.yaml \
  --task-id 1 \
  --certify
```

These checks verify the main Vertex-Softmax primitive: the conservative
decimal-interval reference agrees with high-precision/exhaustive values, and
the scalable PyTorch threshold solver agrees with exhaustive vertex enumeration
on small cases while handling larger score boxes.

## Reviewer Smoke Runs

Small runtime benchmark:

```bash
python benchmark_scalable_vertex_softmax.py \
  --keys 8 16 32 64 \
  --repeats 2 \
  --out results/runtime_smoke.csv
```

Reduced toy attention sweep:

```bash
python attention_cert_method_sweep.py \
  --seq-len 8 \
  --d-in 16 \
  --d-head 16 \
  --trials 100 \
  --batch 50 \
  --epsilons 0.02 \
  --methods ibp wei_lse vertex \
  --out results/toy_sweep_smoke.csv
```

Reduced paper-style scalable sweep:

```bash
python paper_scalable_sweep.py \
  --seq-lens 4 8 16 \
  --seeds 0 \
  --trials 50 \
  --batch 25 \
  --detail-out results/scalable_detail_smoke.csv \
  --summary-out results/scalable_summary_smoke.csv
```

The CROWN-backed benchmark paths in `tiny_vit_block_benchmark.py` require
`auto_LiRPA`. Place an `alpha-beta-CROWN/auto_LiRPA` checkout under
`third_party/` in this folder, or install/adjust imports for your environment.

Reduced full-block smoke run with the end-to-end denominator used for the
non-MNIST full-block paper table:

```bash
python tiny_vit_block_benchmark.py \
  --dataset fashion_mnist \
  --classes 0 1 \
  --out-prefix results/full_block_smoke \
  --train-limit 1024 \
  --eval-limit 64 \
  --cert-limit 32 \
  --cert-denominator eval \
  --block-mode full_mlp \
  --dim 16 \
  --heads 2 \
  --mlp-dim 64 \
  --epochs 1 \
  --epsilons 0.02 \
  --methods CROWN crown_objective_vertex_hybrid \
  --device cuda
```

## Training and Certification Details

The benchmark scripts expose the reported training and verification settings as
command-line arguments. In `tiny_vit_block_benchmark.py`, the default optimizer
is AdamW with `--lr 1e-3`, `--weight-decay 1e-4`, `--batch-size 256`,
`--epochs 8`, and `--grad-clip 1.0`; paper and smoke commands override these
where needed. The same script records class subsets, seeds, patch size, model
dimension, number of heads, MLP width, perturbation radii, certification limits,
PGD attack settings, alpha-CROWN iterations, denominator mode, and the selected
CROWN/Vertex/Hybrid methods in the CSV outputs.

## Compute Notes

The quick checks and synthetic smoke runs can be run on CPU, although PyTorch is
required for the scalable tests. The CROWN-backed image benchmarks require
`auto_LiRPA` and are intended for CUDA-class GPUs. The paper's timing claims use
the RTX 5090 rows reported in the paper. Fashion-MNIST and CIFAR-10 reinforcement
rows were run on mixed accelerator hardware and are used for certificate
tightness, not timing claims. For full paper-number runs, estimate wall time from
the reported per-image or per-trial timings multiplied by the number of certified
examples, methods, perturbation radii, and seeds. Exploratory pilot runs and
failed jobs are not required to reproduce the reported tables and are not included
as paper-number experiments.

## Assets and Licenses

This package redistributes only the anonymized source code in this directory. It
does not redistribute raw datasets, generated results, pretrained models, or
third-party verifier checkouts. Users should comply with upstream dataset and
dependency terms.

| Asset | Use | Source | License / terms note |
| --- | --- | --- | --- |
| MNIST | Downloaded at run time for handwritten-digit experiments. | <https://keras.io/api/datasets/mnist/> and <http://yann.lecun.com/exdb/mnist/> | Keras documents MNIST as CC BY-SA 3.0, with copyright held by Yann LeCun and Corinna Cortes. |
| Fashion-MNIST | Downloaded at run time for non-MNIST grayscale experiments. | <https://github.com/zalandoresearch/fashion-mnist> | MIT license in the upstream Fashion-MNIST repository. |
| CIFAR-10 | Downloaded at run time for grayscale/color CIFAR experiments. | <https://www.cs.toronto.edu/~kriz/cifar.html> and <https://archive.ics.uci.edu/dataset/691/cifar+10> | UCI metadata lists CIFAR-10 under CC BY 4.0; users should check upstream terms before redistribution. |
| PyTorch | Required runtime dependency. | <https://pypi.org/project/torch/> | BSD-style license. |
| NumPy | Required runtime dependency. | <https://numpy.org/about> | Modified BSD license. |
| pandas | Figure/table helper dependency. | <https://pypi.org/project/pandas/> | BSD 3-Clause license. |
| Matplotlib | Figure helper dependency. | <https://matplotlib.org/stable/project/license.html> | BSD-compatible, PSF-based Matplotlib license. |
| alpha-beta-CROWN / auto_LiRPA | Optional external verifier dependency for CROWN-backed paths. | <https://github.com/Verified-Intelligence/alpha-beta-CROWN> | Not bundled; install from upstream and follow the upstream license and dependency terms. |

## Notes

This release does not redistribute raw datasets, pretrained models, or
third-party verifier checkouts. Dataset scripts download MNIST, Fashion-MNIST,
and CIFAR-10 from public sources at run time.

The optional `alpha-beta-CROWN/auto_LiRPA` dependency is not bundled. If used,
install it from its upstream repository.

Full paper-number runs use larger settings and suitable accelerator hardware.
`--cert-denominator clean` reports rates conditional on clean-correct examples;
`--cert-denominator eval` reports end-to-end certified accuracy over the listed
evaluation prefix, counting clean-incorrect examples as uncertified.

The figure/table helpers consume CSV files produced by the sweep scripts. Since
generated results are intentionally excluded, run the relevant sweeps before
calling `generate_paper_figures.py` or `make_runtime_breakdown_table.py`.
