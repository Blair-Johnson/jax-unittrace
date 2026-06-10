from fractions import Fraction
import pytest

from jax_unittrace import ONE, Unit, dimensionless, unit, units


def test_unit_algebra_normalizes_reciprocals():
    m = unit("m")
    s = unit("s")

    assert m / s**2 == units({"m": 1, "s": -2})
    assert (m / s) * (s / m) == ONE
    assert (m**2) ** Fraction(1, 2) == m


def test_dimensionless_spellings():
    assert dimensionless() == ONE
    assert Unit.from_terms("m", ("m", -1)) == ONE


def test_unit_powers_are_immutable_and_hash_stable():
    m = unit("m")
    before = hash(m)

    with pytest.raises(TypeError):
        m.powers["m"] = 2

    assert hash(m) == before
    assert m == unit("m")
