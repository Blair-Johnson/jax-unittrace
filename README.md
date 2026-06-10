# jax-unittrace

A small static unit tracer for ordinary JAX programs.

The tracer uses `jax.make_jaxpr` to inspect a computation graph. Users tag boundary 
arrays with symbolic units, and the library infers intermediate units and flags unit violations.

```python
import jax.numpy as jnp
from jax_unittrace import tag, trace_units, unit

m = unit("m")
s = unit("s")

def velocity(distance, time):
    return distance / time

result = trace_units(
    velocity,
    tag(jnp.ones(8), m),
    tag(jnp.ones(8), s),
)
assert result.output_specs[0].unit == m / s
```

## Unit syntax

Units are arbitrary symbolic atoms with rational exponents.  There is no SI
registry and no conversion table.

```python
m = unit("m")
s = unit("s")
acceleration = m / s**2
also_acceleration = units({"m": 1, "s": -2})
```

## Exponential and logarithmic functions

`jnp.log` (natural log / ln) and friends create derived symbolic units rather
than requiring dimensionless inputs. This matches workflows where `log(m)` is a
meaningful derived quantity:

```python
m = unit("m")
s = unit("s")

trace_units(lambda x: jnp.log(x), tag(jnp.ones(3), m)).output_specs[0].unit
# log[m]

trace_units(lambda x: jnp.exp(x), tag(jnp.ones(3), m)).output_specs[0].unit
# exp[m]

trace_units(lambda x: jnp.exp(jnp.log(x)), tag(jnp.ones(3), m)).output_specs[0].unit
# m
```

Log-derived units are additive in exponent space:

```python
trace_units(
    lambda x, y: jnp.log(x) + jnp.log(y),
    tag(jnp.ones(3), m),
    tag(jnp.ones(3), s),
).output_specs[0].unit
# log[m*s]

trace_units(
    lambda u, x: jnp.exp(u + jnp.log(x)),
    tag(jnp.ones(3), ONE),
    tag(jnp.ones(3), m),
).output_specs[0].unit
# m
```

Other transcendental functions such as trigonometric functions remain strict and
require dimensionless inputs.

## Checking additions

Addition, subtraction, comparisons, and additive reductions require matching
symbolic units.  Mismatches are recorded as diagnostics:

```python
def bad(distance, time):
    return distance + time

result = trace_units(bad, tag(jnp.ones(3), m), tag(jnp.ones(3), s))
assert not result.ok
print(result.errors[0])
```

## Human-readable reports

`trace_units` returns structured data, but `TraceResult` can also format itself
for interactive debugging. The default report emphasizes arrays, units, shapes,
dtypes, diagnostics, and partitioned axes:

```python
def join(distance, time):
    return jnp.concatenate([distance, time], axis=0)

result = trace_units(join, tag(jnp.ones(2), m), tag(jnp.ones(3), s))
print(result.format())
```

Example output:

```text
jax_unittrace report
--------------------------------------------------------------------------------
status: OK
summary: 2 input(s), 1 output(s), 1 equation(s), 0 error(s), 0 warning(s)

Inputs
- input[0]: float32[2]
    unit: m
- input[1]: float32[3]
    unit: s

Outputs
- output[0]: float32[5]
    unit: partitioned
    axis 0 is partitioned:
      axis 0 [0:2)  m
      axis 0 [2:5)  s

Diagnostics
- none
```

For locating an error in the JAX graph, use `debug=True`:

```python
mixed = trace_units(
    lambda x: jnp.sum(x, axis=1),
    tag(jnp.ones((4, 5)), axes={1: [(0, 3, m), (3, 5, s)]}),
)
print(mixed.format(debug=True))
```

The debug report includes the JAXPR equation where the issue arose plus a
unit-annotated version of that equation:

```text
Diagnostics
- [error] partitioned-reduction-mismatch in reduce_sum at equation 0:
    reduction over a partitioned axis would add values with different units

Debug context
- reduce_sum at equation #0: reduction over a partitioned axis would add values with different units
    JAXPR location:
      a:f32[4] = reduce_sum[axes=(1,) out_sharding=None] b
    Unit-annotated location:
      #00 out0:float32[4]{1} = reduce_sum[units] in0:float32[4,5]{1; axis 1: [0:3)=m, [3:5)=s}
```

Convenience methods are available for scripts and notebooks:

```python
result.print_report(debug=True, color=True)
result.save_report("unit-report.txt", debug=True)
```

## Axis partitions

Axis partitions represent piecewise units along array axes, allowing unit tracking across 
concatenated arrays of different units:

```python
def join(a, b):
    return jnp.concatenate([a, b], axis=0)

result = trace_units(join, tag(jnp.ones(2), m), tag(jnp.ones(3), s))
print(result.output_specs[0].describe())
# 1; axis 0: [0:2)=m, [2:5)=s
```

Partition metadata propagates through many shape-only operations and through
multiplication/division by uniform units.  Operations that would require a much
richer region model conservatively emit warnings and keep the uniform fallback
unit.

Nested JAXPRs from common transformations/control-flow primitives such as
`jax.jit`, `lax.cond`, `lax.scan`, and `jnp.where` are also interpreted so unit
errors inside them are reported statically.

## Development

```bash
pixi run test
```
