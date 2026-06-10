"""Static unit tracing for ordinary JAX programs.

Typical use::

    import jax.numpy as jnp
    from jax_unittrace import trace_units, tag, unit

    m = unit("m")
    s = unit("s")

    def velocity(distance, time):
        return distance / time

    result = trace_units(velocity, tag(jnp.ones(3), m), tag(jnp.ones(3), s))
    assert result.output_specs[0].unit == m / s
"""

from .result import Diagnostic, EquationTrace, TraceResult, UnitTraceError
from .specs import ArraySpec, AxisPartition, AxisSegment
from .tracer import TaggedArray, spec, tag, trace, trace_units
from .units import ONE, Unit, as_unit, dimensionless, unit, units

__all__ = [
    "ArraySpec",
    "AxisPartition",
    "AxisSegment",
    "Diagnostic",
    "EquationTrace",
    "ONE",
    "TaggedArray",
    "TraceResult",
    "Unit",
    "UnitTraceError",
    "as_unit",
    "dimensionless",
    "spec",
    "tag",
    "trace",
    "trace_units",
    "unit",
    "units",
]
