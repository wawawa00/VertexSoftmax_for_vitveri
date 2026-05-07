import random
from decimal import Decimal

from interval_vertex_softmax import (
    exhaustive_value_decimal,
    threshold_interval,
    threshold_value_decimal,
)


def _dec(value: float) -> Decimal:
    return Decimal(f"{value:.8f}")


def test_interval_encloses_threshold_value() -> None:
    rng = random.Random(0)
    for keys in range(1, 9):
        for _ in range(40):
            c = [_dec(rng.uniform(-3.0, 3.0)) for _ in range(keys)]
            center = [rng.uniform(-5.0, 5.0) for _ in range(keys)]
            radius = [rng.uniform(0.0, 4.0) for _ in range(keys)]
            ell = [_dec(a - b) for a, b in zip(center, radius)]
            u = [_dec(a + b) for a, b in zip(center, radius)]
            interval = threshold_interval(c, ell, u, precision=70, guard_digits=30)
            point = threshold_value_decimal(c, ell, u, precision=120)
            exhaustive = exhaustive_value_decimal(c, ell, u, precision=120)
            assert interval.lower <= point <= interval.upper
            assert interval.lower <= exhaustive <= interval.upper
            assert abs(point - exhaustive) <= Decimal("1e-95")


def test_large_dynamic_range() -> None:
    c = [Decimal("-2.0"), Decimal("0.5"), Decimal("3.0"), Decimal("-0.1")]
    ell = [Decimal("-1000"), Decimal("-850"), Decimal("-1200"), Decimal("-900")]
    u = [Decimal("-940"), Decimal("-830"), Decimal("-1190"), Decimal("-860")]
    interval = threshold_interval(c, ell, u, precision=80, guard_digits=40)
    point = threshold_value_decimal(c, ell, u, precision=140)
    assert interval.lower <= point <= interval.upper
    assert interval.width >= 0


def main() -> None:
    test_interval_encloses_threshold_value()
    test_large_dynamic_range()
    print("interval vertex-softmax tests passed")


if __name__ == "__main__":
    main()
