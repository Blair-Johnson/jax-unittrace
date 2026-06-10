"""Symbolic unit algebra for jax_unittrace.

Units are deliberately *not* tied to SI.  A unit is just a normalized
mapping from user-provided atom names to rational exponents.  This is enough
to represent values such as ``m / s**2`` while keeping the API open-ended.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral, Rational
from types import MappingProxyType
from typing import Iterable, Mapping, Union

Exponent = Union[int, Fraction]
UnitLike = Union["Unit", str, Mapping[str, Exponent], None]


@dataclass(frozen=True)
class Unit:
    """A symbolic product of arbitrary unit atoms raised to powers.

    Examples
    --------
    >>> m = Unit.atom("m")
    >>> s = Unit.atom("s")
    >>> m / s**2
    m*s^-2
    """

    powers: Mapping[str, Fraction]

    def __post_init__(self) -> None:
        normalized: dict[str, Fraction] = {}
        for name, exponent in self.powers.items():
            if not isinstance(name, str) or not name:
                raise ValueError("unit atom names must be non-empty strings")
            exp = _to_fraction(exponent)
            if exp:
                normalized[name] = exp
        object.__setattr__(self, "powers", MappingProxyType(dict(sorted(normalized.items()))))

    @classmethod
    def dimensionless(cls) -> "Unit":
        return cls({})

    @classmethod
    def atom(cls, name: str) -> "Unit":
        return cls({name: Fraction(1)})

    @classmethod
    def from_terms(cls, *terms: str | tuple[str, Exponent]) -> "Unit":
        """Build a unit from atom names and/or ``(name, exponent)`` tuples."""

        powers: dict[str, Fraction] = {}
        for term in terms:
            if isinstance(term, str):
                name, exponent = term, Fraction(1)
            else:
                name, exponent = term
            powers[name] = powers.get(name, Fraction(0)) + _to_fraction(exponent)
        return cls(powers)

    @property
    def is_dimensionless(self) -> bool:
        return not self.powers

    def __mul__(self, other: UnitLike) -> "Unit":
        other_unit = as_unit(other)
        powers = dict(self.powers)
        for name, exponent in other_unit.powers.items():
            powers[name] = powers.get(name, Fraction(0)) + exponent
        return Unit(powers)

    def __rmul__(self, other: UnitLike) -> "Unit":
        return as_unit(other) * self

    def __truediv__(self, other: UnitLike) -> "Unit":
        return self * (as_unit(other) ** -1)

    def __rtruediv__(self, other: UnitLike) -> "Unit":
        return as_unit(other) / self

    def __pow__(self, exponent: Exponent) -> "Unit":
        exp = _to_fraction(exponent)
        return Unit({name: power * exp for name, power in self.powers.items()})

    def reciprocal(self) -> "Unit":
        return self ** -1

    def __bool__(self) -> bool:
        return not self.is_dimensionless

    def __str__(self) -> str:
        if self.is_dimensionless:
            return "1"
        pieces: list[str] = []
        for name, exponent in self.powers.items():
            if exponent == 1:
                pieces.append(name)
            else:
                pieces.append(f"{name}^{_format_fraction(exponent)}")
        return "*".join(pieces)

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash(tuple(self.powers.items()))

def unit(name: str) -> Unit:
    """Return an atomic unit with the given arbitrary name."""

    return Unit.atom(name)


def units(mapping: Mapping[str, Exponent]) -> Unit:
    """Return a unit from a ``{name: exponent}`` mapping."""

    return Unit(mapping)


def dimensionless() -> Unit:
    return Unit.dimensionless()


ONE = Unit.dimensionless()


def as_unit(value: UnitLike) -> Unit:
    """Coerce user-facing unit inputs into :class:`Unit`."""

    if value is None:
        return ONE
    if isinstance(value, Unit):
        return value
    if isinstance(value, str):
        if value in {"", "1", "dimensionless"}:
            return ONE
        return Unit.atom(value)
    if isinstance(value, Mapping):
        return Unit(value)
    raise TypeError(f"cannot interpret {value!r} as a Unit")


def _to_fraction(value: Exponent) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, Integral):
        return Fraction(int(value), 1)
    if isinstance(value, Rational):
        return Fraction(value)
    raise TypeError(f"unit exponents must be rational, got {type(value).__name__}")


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"
