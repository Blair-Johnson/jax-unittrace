import jax
import jax.numpy as jnp
from jax import lax
import pytest

from jax_unittrace import ONE, tag, trace_units, unit


def test_traces_existing_program_without_wrapping_arrays():
    m = unit("m")
    s = unit("s")

    def velocity(distance, time):
        # The function sees ordinary JAX tracers, not TaggedArray instances.
        assert not hasattr(distance, "spec")
        return distance / time

    result = trace_units(velocity, tag(jnp.ones(4), m), tag(jnp.ones(4), s))

    assert result.ok
    assert result.output_specs[0].unit == m / s


def test_addition_with_same_units_is_ok():
    m = unit("m")

    def add(a, b):
        return a + b

    result = trace_units(add, tag(jnp.ones(2), m), tag(jnp.ones(2), m))

    assert result.ok
    assert result.output_specs[0].unit == m


def test_addition_with_different_units_is_an_error():
    m = unit("m")
    s = unit("s")

    def bad(a, b):
        return a + b

    result = trace_units(bad, tag(jnp.ones(2), m), tag(jnp.ones(2), s))

    assert not result.ok
    assert result.errors[0].code == "unit-mismatch"
    assert "cannot add values" in result.errors[0].message
    with pytest.raises(ValueError):
        result.raise_on_error()


def test_exp_and_natural_log_of_dimensionless_return_dimensionless():
    result = trace_units(lambda x: jnp.exp(x) + jnp.log(x), tag(jnp.ones(2), ONE))

    assert result.ok
    assert result.output_specs[0].unit == ONE


def test_log_of_unitful_value_creates_derived_log_unit():
    m = unit("m")

    result = trace_units(lambda x: jnp.log(x), tag(jnp.ones(2), m))

    assert result.ok
    assert result.output_specs[0].unit == unit("log[m]")


def test_exp_of_unitful_value_creates_derived_exp_unit():
    m = unit("m")

    result = trace_units(lambda x: jnp.exp(x), tag(jnp.ones(2), m))

    assert result.ok
    assert result.output_specs[0].unit == unit("exp[m]")


def test_exp_of_log_unit_round_trips_original_unit():
    m = unit("m")

    result = trace_units(lambda x: jnp.exp(jnp.log(x)), tag(jnp.ones(2), m))

    assert result.ok
    assert result.output_specs[0].unit == m


def test_exp_of_dimensionless_plus_log_unit_round_trips_log_component():
    m = unit("m")

    result = trace_units(lambda u, x: jnp.exp(u + jnp.log(x)), tag(jnp.ones(2), ONE), tag(jnp.ones(2), m))

    assert result.ok
    assert result.output_specs[0].unit == m


def test_log_units_add_multiplicatively_in_exponent_space():
    m = unit("m")
    s = unit("s")

    result = trace_units(lambda x, y: jnp.log(x) + jnp.log(y), tag(jnp.ones(2), m), tag(jnp.ones(2), s))

    assert result.ok
    assert result.output_specs[0].unit == unit("log[m*s]")


def test_log_unit_plus_raw_unit_still_errors():
    m = unit("m")

    result = trace_units(lambda x, y: jnp.log(x) + y, tag(jnp.ones(2), m), tag(jnp.ones(2), m))

    assert not result.ok
    assert result.errors[0].code == "unit-mismatch"


def test_dot_general_multiplies_units():
    m = unit("m")
    kg = unit("kg")

    def f(a, b):
        return a @ b

    result = trace_units(f, tag(jnp.ones((2, 3)), kg), tag(jnp.ones((3, 4)), m))

    assert result.ok
    assert result.output_specs[0].unit == kg * m
    assert result.output_specs[0].shape == (2, 4)


def test_literal_power_raises_units_to_exponent():
    m = unit("m")

    def area_side(x):
        return x ** 2.0

    result = trace_units(area_side, tag(jnp.ones(2), m))

    assert result.ok
    assert result.output_specs[0].unit == m**2


def test_square_mean_and_max_preserve_expected_units():
    m = unit("m")

    square_result = trace_units(lambda x: jnp.square(x), tag(jnp.ones(3), m))
    mean_result = trace_units(lambda x: jnp.mean(x), tag(jnp.ones(3), m))
    max_result = trace_units(lambda x: jnp.max(x), tag(jnp.ones(3), m))

    assert square_result.ok
    assert square_result.output_specs[0].unit == m**2
    assert mean_result.ok
    assert mean_result.output_specs[0].unit == m
    assert max_result.ok
    assert max_result.output_specs[0].unit == m


def test_elementwise_maximum_preserves_compatible_units():
    m = unit("m")

    result = trace_units(lambda a, b: jnp.maximum(a, b), tag(jnp.ones(3), m), tag(jnp.ones(3), m))

    assert result.ok
    assert result.output_specs[0].unit == m


def test_elementwise_minimum_flags_mismatched_units():
    m = unit("m")
    s = unit("s")

    result = trace_units(lambda a, b: jnp.minimum(a, b), tag(jnp.ones(3), m), tag(jnp.ones(3), s))

    assert not result.ok
    assert result.errors[0].code == "unit-mismatch"


def test_adding_dimensioned_value_to_literal_is_mismatch():
    m = unit("m")

    result = trace_units(lambda x: x + 1.0, tag(jnp.ones(3), m))

    assert not result.ok
    assert result.errors[0].code == "unit-mismatch"


def test_jitted_function_is_traced_through_nested_jaxpr():
    m = unit("m")
    s = unit("s")

    @jax.jit
    def bad(a, b):
        return a + b

    result = trace_units(bad, tag(jnp.ones(2), m), tag(jnp.ones(2), s))

    assert not result.ok
    assert result.errors[0].code == "unit-mismatch"


def test_where_requires_compatible_branch_units():
    m = unit("m")
    s = unit("s")

    def f(a, b):
        return jnp.where(a > 0, a, b)

    result = trace_units(f, tag(jnp.ones(2), m), tag(jnp.ones(2), s))

    assert not result.ok
    assert result.errors[0].code == "unit-mismatch"


def test_cond_requires_compatible_branch_output_units():
    m = unit("m")
    s = unit("s")

    def f(pred, x, y):
        return lax.cond(pred, lambda pair: pair[0], lambda pair: pair[1], (x, y))

    result = trace_units(f, True, tag(jnp.ones(2), m), tag(jnp.ones(2), s))

    assert not result.ok
    assert result.errors[0].code == "unit-mismatch"
    assert result.errors[0].primitive == "cond"


def test_scan_body_checks_units_and_propagates_outputs():
    m = unit("m")

    def f(init, xs):
        return lax.scan(lambda carry, x: (carry + x, carry * x), init, xs)

    result = trace_units(f, tag(jnp.ones(2), m), tag(jnp.ones((3, 2)), m))

    assert result.ok
    carry_out, ys_out = result.output_specs
    assert carry_out.unit == m
    assert ys_out.unit == m**2
    assert ys_out.shape == (3, 2)


def test_scan_flags_carry_unit_mismatch():
    m = unit("m")
    s = unit("s")

    def f(init, xs):
        return lax.scan(lambda carry, x: (carry + x, carry), init, xs)

    result = trace_units(f, tag(jnp.ones(2), m), tag(jnp.ones((3, 2)), s))

    assert not result.ok
    assert result.errors[0].code == "unit-mismatch"
