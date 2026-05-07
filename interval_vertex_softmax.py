#!/usr/bin/env python3
"""Conservative reference evaluator for the Vertex-Softmax threshold primitive.

The optimized experiments use the vectorized PyTorch implementation in
``attention_cert_method_sweep.py`` and ``tiny_vit_*_benchmark.py``.  This file is
not meant to be fast.  It is a small, dependency-free reference showing how the
same K+1 threshold candidates can be evaluated with outward-enclosed decimal
intervals.

The returned lower endpoint is conservative for the score-box subproblem under
the supplied decimal inputs: it is intentionally allowed to sit slightly below
the real-arithmetic optimum.  A production proof-carrying verifier should use
directed-rounding interval arithmetic for all upstream quantities as well.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from typing import Iterable


@dataclass(frozen=True)
class Interval:
    lo: Decimal
    hi: Decimal

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(f"invalid interval [{self.lo}, {self.hi}]")


@dataclass(frozen=True)
class CandidateInterval:
    threshold: int
    value: Interval


@dataclass(frozen=True)
class ThresholdIntervalResult:
    lower: Decimal
    upper: Decimal
    lower_threshold: int
    upper_threshold: int
    candidates: tuple[CandidateInterval, ...]

    @property
    def width(self) -> Decimal:
        return self.upper - self.lower


def _decimal(value: object) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(repr(value))
    return Decimal(str(value))


def _ctx(precision: int, rounding: str):
    ctx = localcontext()
    context = ctx.__enter__()
    context.prec = precision
    context.rounding = rounding
    context.Emin = -999999999
    context.Emax = 999999999
    return ctx, context


def _round_down(value: Decimal, precision: int) -> Decimal:
    manager, _context = _ctx(precision, ROUND_FLOOR)
    with manager:
        return +value


def _round_up(value: Decimal, precision: int) -> Decimal:
    manager, _context = _ctx(precision, ROUND_CEILING)
    with manager:
        return +value


def _add_down(a: Decimal, b: Decimal, precision: int) -> Decimal:
    manager, _context = _ctx(precision, ROUND_FLOOR)
    with manager:
        return +(a + b)


def _add_up(a: Decimal, b: Decimal, precision: int) -> Decimal:
    manager, _context = _ctx(precision, ROUND_CEILING)
    with manager:
        return +(a + b)


def _mul_down(a: Decimal, b: Decimal, precision: int) -> Decimal:
    manager, _context = _ctx(precision, ROUND_FLOOR)
    with manager:
        return +(a * b)


def _mul_up(a: Decimal, b: Decimal, precision: int) -> Decimal:
    manager, _context = _ctx(precision, ROUND_CEILING)
    with manager:
        return +(a * b)


def _div_down(a: Decimal, b: Decimal, precision: int) -> Decimal:
    manager, _context = _ctx(precision, ROUND_FLOOR)
    with manager:
        return +(a / b)


def _div_up(a: Decimal, b: Decimal, precision: int) -> Decimal:
    manager, _context = _ctx(precision, ROUND_CEILING)
    with manager:
        return +(a / b)


def _next_down(value: Decimal, precision: int, steps: int) -> Decimal:
    manager, context = _ctx(precision, ROUND_FLOOR)
    with manager:
        out = +value
        for _ in range(steps):
            out = out.next_minus(context)
        return out


def _next_up(value: Decimal, precision: int, steps: int) -> Decimal:
    manager, context = _ctx(precision, ROUND_CEILING)
    with manager:
        out = +value
        for _ in range(steps):
            out = out.next_plus(context)
        return out


def _exp_interval(
    x: Decimal,
    precision: int,
    guard_digits: int,
    outward_steps: int,
) -> Interval:
    guard_precision = precision + guard_digits
    with localcontext() as context:
        context.prec = guard_precision
        context.Emin = -999999999
        context.Emax = 999999999
        midpoint = x.exp(context)
    lo = _next_down(_round_down(midpoint, precision), precision, outward_steps)
    hi = _next_up(_round_up(midpoint, precision), precision, outward_steps)
    return Interval(lo, hi)


def _add_interval(a: Interval, b: Interval, precision: int) -> Interval:
    return Interval(
        _add_down(a.lo, b.lo, precision),
        _add_up(a.hi, b.hi, precision),
    )


def _scale_interval(a: Interval, coefficient: Decimal, precision: int) -> Interval:
    if coefficient >= 0:
        return Interval(
            _mul_down(coefficient, a.lo, precision),
            _mul_up(coefficient, a.hi, precision),
        )
    return Interval(
        _mul_down(coefficient, a.hi, precision),
        _mul_up(coefficient, a.lo, precision),
    )


def _divide_interval(numerator: Interval, denominator: Interval, precision: int) -> Interval:
    if denominator.lo <= 0:
        raise ValueError(f"denominator interval must be positive, got {denominator}")
    lows = []
    highs = []
    for n in (numerator.lo, numerator.hi):
        for d in (denominator.lo, denominator.hi):
            lows.append(_div_down(n, d, precision))
            highs.append(_div_up(n, d, precision))
    return Interval(min(lows), max(highs))


def threshold_interval(
    c: Iterable[object],
    ell: Iterable[object],
    u: Iterable[object],
    *,
    precision: int = 80,
    guard_digits: int = 30,
    outward_steps: int = 4,
) -> ThresholdIntervalResult:
    """Conservatively enclose min_s c^T softmax(s) over ell <= s <= u.

    The lower endpoint of the returned result is the value to use as a sound
    lower bound for the score-box subproblem, assuming the supplied inputs are
    themselves sound decimal quantities.
    """
    coeffs = tuple(_decimal(v) for v in c)
    lows = tuple(_decimal(v) for v in ell)
    highs = tuple(_decimal(v) for v in u)
    if not (len(coeffs) == len(lows) == len(highs)):
        raise ValueError("c, ell, and u must have the same length")
    if not coeffs:
        raise ValueError("at least one coordinate is required")
    for low, high in zip(lows, highs):
        if low > high:
            raise ValueError(f"lower bound {low} exceeds upper bound {high}")

    shift = max(highs)
    exp_l = tuple(
        _exp_interval(low - shift, precision, guard_digits, outward_steps) for low in lows
    )
    exp_u = tuple(
        _exp_interval(high - shift, precision, guard_digits, outward_steps) for high in highs
    )
    order = sorted(range(len(coeffs)), key=lambda idx: coeffs[idx])
    sorted_coeffs = tuple(coeffs[idx] for idx in order)
    sorted_exp_l = tuple(exp_l[idx] for idx in order)
    sorted_exp_u = tuple(exp_u[idx] for idx in order)

    candidates: list[CandidateInterval] = []
    zero = Interval(Decimal(0), Decimal(0))
    for threshold in range(len(coeffs) + 1):
        numerator = zero
        denominator = zero
        for pos, coefficient in enumerate(sorted_coeffs):
            y = sorted_exp_u[pos] if pos < threshold else sorted_exp_l[pos]
            numerator = _add_interval(
                numerator,
                _scale_interval(y, coefficient, precision),
                precision,
            )
            denominator = _add_interval(denominator, y, precision)
        value = _divide_interval(numerator, denominator, precision)
        candidates.append(CandidateInterval(threshold, value))

    lower_candidate = min(candidates, key=lambda item: item.value.lo)
    upper_candidate = min(candidates, key=lambda item: item.value.hi)
    return ThresholdIntervalResult(
        lower=lower_candidate.value.lo,
        upper=upper_candidate.value.hi,
        lower_threshold=lower_candidate.threshold,
        upper_threshold=upper_candidate.threshold,
        candidates=tuple(candidates),
    )


def threshold_value_decimal(
    c: Iterable[object],
    ell: Iterable[object],
    u: Iterable[object],
    *,
    precision: int = 120,
) -> Decimal:
    """High-precision point evaluation of the K+1 threshold solver."""
    coeffs = tuple(_decimal(v) for v in c)
    lows = tuple(_decimal(v) for v in ell)
    highs = tuple(_decimal(v) for v in u)
    shift = max(highs)
    order = sorted(range(len(coeffs)), key=lambda idx: coeffs[idx])
    with localcontext() as context:
        context.prec = precision
        context.Emin = -999999999
        context.Emax = 999999999
        exp_l = tuple((low - shift).exp(context) for low in lows)
        exp_u = tuple((high - shift).exp(context) for high in highs)
        best: Decimal | None = None
        for threshold in range(len(coeffs) + 1):
            numerator = Decimal(0)
            denominator = Decimal(0)
            for pos, idx in enumerate(order):
                y = exp_u[idx] if pos < threshold else exp_l[idx]
                numerator += coeffs[idx] * y
                denominator += y
            value = numerator / denominator
            if best is None or value < best:
                best = value
    if best is None:
        raise ValueError("at least one coordinate is required")
    return best


def exhaustive_value_decimal(
    c: Iterable[object],
    ell: Iterable[object],
    u: Iterable[object],
    *,
    precision: int = 120,
) -> Decimal:
    """High-precision exhaustive vertex evaluation for small K regression tests."""
    coeffs = tuple(_decimal(v) for v in c)
    lows = tuple(_decimal(v) for v in ell)
    highs = tuple(_decimal(v) for v in u)
    shift = max(highs)
    with localcontext() as context:
        context.prec = precision
        context.Emin = -999999999
        context.Emax = 999999999
        exp_l = tuple((low - shift).exp(context) for low in lows)
        exp_u = tuple((high - shift).exp(context) for high in highs)
        best: Decimal | None = None
        for mask in itertools.product((False, True), repeat=len(coeffs)):
            numerator = Decimal(0)
            denominator = Decimal(0)
            for idx, use_upper in enumerate(mask):
                y = exp_u[idx] if use_upper else exp_l[idx]
                numerator += coeffs[idx] * y
                denominator += y
            value = numerator / denominator
            if best is None or value < best:
                best = value
    if best is None:
        raise ValueError("at least one coordinate is required")
    return best


def _parse_json_list(raw: str) -> list[object]:
    value = json.loads(raw)
    if not isinstance(value, list):
        raise argparse.ArgumentTypeError("expected a JSON list")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=_parse_json_list, default=[0.4, -1.0, 2.0, 0.1])
    parser.add_argument("--ell", type=_parse_json_list, default=[-2.0, -1.5, -3.0, -0.2])
    parser.add_argument("--u", type=_parse_json_list, default=[0.5, 1.0, -0.5, 0.7])
    parser.add_argument("--precision", type=int, default=80)
    parser.add_argument("--guard-digits", type=int, default=30)
    args = parser.parse_args()

    result = threshold_interval(
        args.c,
        args.ell,
        args.u,
        precision=args.precision,
        guard_digits=args.guard_digits,
    )
    point = threshold_value_decimal(args.c, args.ell, args.u, precision=args.precision + 40)
    print(f"conservative_lower={result.lower}")
    print(f"conservative_upper={result.upper}")
    print(f"interval_width={result.width}")
    print(f"best_lower_threshold={result.lower_threshold}")
    print(f"best_upper_threshold={result.upper_threshold}")
    print(f"high_precision_point={point}")


if __name__ == "__main__":
    main()
