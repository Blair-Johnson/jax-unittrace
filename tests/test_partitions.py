import jax.numpy as jnp

from jax_unittrace import ONE, tag, trace_units, unit


def test_concatenate_different_units_creates_axis_partition():
    m = unit("m")
    s = unit("s")

    def join(a, b):
        return jnp.concatenate([a, b], axis=0)

    result = trace_units(join, tag(jnp.ones(2), m), tag(jnp.ones(3), s))

    assert result.ok
    out = result.output_specs[0]
    assert out.shape == (5,)
    assert out.unit == ONE
    assert out.describe().startswith("1; axis 0:")
    assert len(out.partitions) == 1
    partition = out.partitions[0]
    assert partition.axis == 0
    assert [(seg.start, seg.stop, seg.unit) for seg in partition.segments] == [
        (0, 2, m),
        (2, 5, s),
    ]


def test_partition_units_propagate_through_multiplication():
    m = unit("m")
    s = unit("s")
    kg = unit("kg")

    def join_and_scale(a, b, mass):
        return jnp.concatenate([a, b], axis=0) * mass

    result = trace_units(
        join_and_scale,
        tag(jnp.ones(2), m),
        tag(jnp.ones(3), s),
        tag(jnp.ones(5), kg),
    )

    assert result.ok
    partition = result.output_specs[0].partitions[0]
    assert [(seg.start, seg.stop, seg.unit) for seg in partition.segments] == [
        (0, 2, m * kg),
        (2, 5, s * kg),
    ]


def test_adding_partitioned_array_to_uniform_array_flags_segment_mismatch():
    m = unit("m")
    s = unit("s")

    def f(a, b, reference):
        return jnp.concatenate([a, b], axis=0) + reference

    result = trace_units(
        f,
        tag(jnp.ones(2), m),
        tag(jnp.ones(3), s),
        tag(jnp.ones(5), m),
    )

    assert not result.ok
    assert result.errors[0].code == "unit-mismatch"


def test_manual_axis_partition_can_be_tagged_and_sliced():
    m = unit("m")
    s = unit("s")

    def f(x):
        return x[1:4]

    result = trace_units(
        f,
        tag(jnp.ones(5), axes={0: [(0, 2, m), (2, 5, s)]}),
    )

    assert result.ok
    partition = result.output_specs[0].partitions[0]
    assert [(seg.start, seg.stop, seg.unit) for seg in partition.segments] == [
        (0, 1, m),
        (1, 3, s),
    ]


def test_same_layout_partitioned_multiplication_combines_segment_units():
    m = unit("m")
    s = unit("s")

    def f(x):
        return x * x

    result = trace_units(
        f,
        tag(jnp.ones(5), axes={0: [(0, 2, m), (2, 5, s)]}),
    )

    assert result.ok
    partition = result.output_specs[0].partitions[0]
    assert [(seg.start, seg.stop, seg.unit) for seg in partition.segments] == [
        (0, 2, m**2),
        (2, 5, s**2),
    ]


def test_same_layout_partitioned_division_combines_segment_units():
    m = unit("m")
    s = unit("s")

    def f(x):
        return x / x

    result = trace_units(
        f,
        tag(jnp.ones(5), axes={0: [(0, 2, m), (2, 5, s)]}),
    )

    assert result.ok
    partition = result.output_specs[0].partitions[0]
    assert [(seg.start, seg.stop, seg.unit) for seg in partition.segments] == [
        (0, 2, m / m),
        (2, 5, s / s),
    ]


def test_sum_over_homogeneous_partitioned_axis_infers_common_unit():
    m = unit("m")

    result = trace_units(
        lambda x: jnp.sum(x),
        tag(jnp.ones(5), axes={0: [(0, 2, m), (2, 5, m)]}),
    )

    assert result.ok
    assert result.output_specs[0].unit == m
    assert result.output_specs[0].partitions == ()


def test_max_over_homogeneous_partitioned_axis_infers_common_unit():
    m = unit("m")

    result = trace_units(
        lambda x: jnp.max(x),
        tag(jnp.ones(5), axes={0: [(0, 2, m), (2, 5, m)]}),
    )

    assert result.ok
    assert result.output_specs[0].unit == m


def test_full_cover_homogeneous_partition_adds_to_matching_uniform_unit():
    m = unit("m")

    def f(partitioned, uniform):
        return partitioned + uniform

    result = trace_units(
        f,
        tag(jnp.ones(5), axes={0: [(0, 2, m), (2, 5, m)]}),
        tag(jnp.ones(5), m),
    )

    assert result.ok
    assert result.output_specs[0].unit == m


def test_partial_partition_gap_uses_fallback_for_addition_compatibility():
    m = unit("m")

    def f(partitioned, uniform):
        return partitioned + uniform

    result = trace_units(
        f,
        tag(jnp.ones(5), axes={0: [(0, 2, m)]}),
        tag(jnp.ones(5), m),
    )

    assert not result.ok
    assert result.errors[0].code == "unit-mismatch"


def test_same_layout_full_cover_partitioned_arrays_add_with_different_fallbacks():
    m = unit("m")

    def f(a, b):
        return a + b

    result = trace_units(
        f,
        tag(jnp.ones(5), axes={0: [(0, 2, m), (2, 5, m)]}),
        tag(jnp.ones(5), "dimensionless", axes={0: [(0, 2, m), (2, 5, m)]}),
    )

    assert result.ok
    assert result.output_specs[0].unit == ONE
    assert result.output_specs[0].partitions[0].segments[0].unit == m


def test_same_layout_partial_partitioned_arrays_require_matching_fallbacks():
    m = unit("m")
    s = unit("s")

    def f(a, b):
        return a + b

    result = trace_units(
        f,
        tag(jnp.ones(5), axes={0: [(0, 2, m)]}),
        tag(jnp.ones(5), s, axes={0: [(0, 2, m)]}),
    )

    assert not result.ok
    assert result.errors[0].code == "unit-mismatch"
