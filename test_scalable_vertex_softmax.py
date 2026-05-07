#!/usr/bin/env python3

import torch

from attention_cert_method_sweep import (
    softmax_box_expectation_bounds,
    softmax_box_expectation_bounds_exhaustive,
    softmax_box_expectation_min,
    vertex_masks,
)


def test_threshold_matches_exhaustive() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    for keys in range(1, 11):
        masks = vertex_masks(keys, device)
        for _ in range(50):
            batch = 3
            queries = 4
            heads = 5
            center = torch.randn(batch, queries, keys, device=device)
            radius = torch.rand(batch, queries, keys, device=device) * 3.0
            score_l = center - radius
            score_u = center + radius
            coeff_l = torch.randn(batch, keys, heads, device=device)
            coeff_u = torch.randn(batch, keys, heads, device=device)

            threshold_l, threshold_u = softmax_box_expectation_bounds(
                score_l, score_u, coeff_l, coeff_u
            )
            exhaustive_l, exhaustive_u = softmax_box_expectation_bounds_exhaustive(
                score_l, score_u, coeff_l, coeff_u, masks, vertex_chunk=64
            )

            torch.testing.assert_close(threshold_l, exhaustive_l, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(threshold_u, exhaustive_u, rtol=2e-5, atol=2e-6)


def test_threshold_large_shapes() -> None:
    torch.manual_seed(1)
    batch = 2
    queries = 3
    keys = 256
    heads = 4
    center = torch.randn(batch, queries, keys)
    radius = torch.rand(batch, queries, keys)
    score_l = center - radius
    score_u = center + radius
    coeff = torch.randn(batch, keys, heads)

    result = softmax_box_expectation_min(score_l, score_u, coeff)

    assert result.shape == (batch, queries, heads)
    assert torch.isfinite(result).all()
    assert result.min() >= coeff.min() - 1e-5
    assert result.max() <= coeff.max() + 1e-5


def main() -> None:
    test_threshold_matches_exhaustive()
    test_threshold_large_shapes()
    print("scalable vertex-softmax tests passed")


if __name__ == "__main__":
    main()

