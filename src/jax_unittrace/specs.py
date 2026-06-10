"""Array-level unit metadata and partition propagation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Callable, Iterable, Mapping, Sequence

from .units import ONE, Unit, UnitLike, as_unit


@dataclass(frozen=True)
class AxisSegment:
    """A half-open interval on one array axis carrying an element unit."""

    start: int
    stop: int
    unit: Unit

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop < self.start:
            raise ValueError("axis segments must satisfy 0 <= start <= stop")
        object.__setattr__(self, "unit", as_unit(self.unit))

    def shifted(self, offset: int) -> "AxisSegment":
        return AxisSegment(self.start + offset, self.stop + offset, self.unit)

    def with_unit(self, unit: UnitLike) -> "AxisSegment":
        return AxisSegment(self.start, self.stop, as_unit(unit))


@dataclass(frozen=True)
class AxisPartition:
    """Piecewise units along a single array axis."""

    axis: int
    segments: tuple[AxisSegment, ...]

    def __post_init__(self) -> None:
        if self.axis < 0:
            raise ValueError("partition axis must be non-negative")
        segments = tuple(sorted(self.segments, key=lambda s: (s.start, s.stop)))
        last_stop = 0
        for segment in segments:
            if segment.start < last_stop:
                raise ValueError("axis partition segments may not overlap")
            last_stop = segment.stop
        object.__setattr__(self, "segments", segments)

    def shifted(self, offset: int) -> "AxisPartition":
        return AxisPartition(self.axis, tuple(s.shifted(offset) for s in self.segments))

    def remap_axis(self, axis: int) -> "AxisPartition":
        return AxisPartition(axis, self.segments)

    def map_units(self, fn: Callable[[Unit], Unit]) -> "AxisPartition":
        return AxisPartition(self.axis, tuple(s.with_unit(fn(s.unit)) for s in self.segments))


PartitionInput = Mapping[int, Sequence[tuple[int, int, UnitLike] | AxisSegment]] | Sequence[AxisPartition]


@dataclass(frozen=True)
class ArraySpec:
    """Unit metadata for a JAX value.

    ``unit`` is the uniform fallback unit.  ``partitions`` optionally refines
    the element unit over half-open intervals on array axes, which lets the
    tracer represent values such as ``concatenate([meters, seconds])``.
    """

    unit: Unit = ONE
    partitions: tuple[AxisPartition, ...] = field(default_factory=tuple)
    shape: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit", as_unit(self.unit))
        object.__setattr__(self, "partitions", _normalize_partitions(self.partitions))
        if self.shape is not None:
            object.__setattr__(self, "shape", tuple(int(x) for x in self.shape))

    @classmethod
    def from_user(
        cls,
        unit: UnitLike = None,
        axes: PartitionInput | None = None,
        shape: Sequence[int] | None = None,
    ) -> "ArraySpec":
        return cls(as_unit(unit), normalize_axes(axes), None if shape is None else tuple(shape))

    @property
    def is_uniform(self) -> bool:
        return not self.partitions

    def with_shape(self, shape: Sequence[int] | None) -> "ArraySpec":
        return ArraySpec(self.unit, self.partitions, None if shape is None else tuple(shape))

    def without_partitions(self) -> "ArraySpec":
        return ArraySpec(self.unit, (), self.shape)

    def equivalent_units(self, other: "ArraySpec") -> bool:
        """Return true when element units are symbolically identical."""

        return self.unit == other.unit and self.partitions == other.partitions

    def all_units(self) -> tuple[Unit, ...]:
        units = [self.unit]
        for partition in self.partitions:
            units.extend(segment.unit for segment in partition.segments)
        return tuple(dict.fromkeys(units))

    def map_units(self, fn: Callable[[Unit], Unit]) -> "ArraySpec":
        return ArraySpec(
            fn(self.unit),
            tuple(partition.map_units(fn) for partition in self.partitions),
            self.shape,
        )

    def multiply(self, other: "ArraySpec") -> "ArraySpec":
        return combine_multiplicative(self, other, lambda a, b: a * b)

    def divide(self, other: "ArraySpec") -> "ArraySpec":
        return combine_multiplicative(self, other, lambda a, b: a / b)

    def power(self, exponent: int | Fraction) -> "ArraySpec":
        return self.map_units(lambda unit: unit**exponent)

    def describe(self) -> str:
        if self.is_uniform:
            return str(self.unit)
        pieces = [str(self.unit)]
        for partition in self.partitions:
            segments = ", ".join(
                f"[{segment.start}:{segment.stop})={segment.unit}" for segment in partition.segments
            )
            pieces.append(f"axis {partition.axis}: {segments}")
        return "; ".join(pieces)


def normalize_axes(axes: PartitionInput | None) -> tuple[AxisPartition, ...]:
    if axes is None:
        return ()
    if isinstance(axes, Mapping):
        partitions: list[AxisPartition] = []
        for axis, entries in axes.items():
            segments: list[AxisSegment] = []
            for entry in entries:
                if isinstance(entry, AxisSegment):
                    segments.append(entry)
                else:
                    start, stop, unit = entry
                    segments.append(AxisSegment(int(start), int(stop), as_unit(unit)))
            partitions.append(AxisPartition(int(axis), tuple(segments)))
        return _normalize_partitions(partitions)
    return _normalize_partitions(tuple(axes))


def compatible_for_addition(left: ArraySpec, right: ArraySpec) -> bool:
    """Strict symbolic compatibility used for add/subtract/compare checks."""

    if left.equivalent_units(right):
        return True
    if left.is_uniform and right.is_uniform:
        return left.unit == right.unit
    # Uniform values may combine with a partitioned array if every element in
    # the partitioned value has that same unit.  Gaps use the fallback unit; a
    # fully-covered partition can therefore have a neutral fallback.
    if left.is_uniform:
        return _all_elements_have_unit(right, left.unit)
    if right.is_uniform:
        return _all_elements_have_unit(left, right.unit)
    return _partitioned_specs_compatible_for_addition(left, right)


def _all_elements_have_unit(spec: ArraySpec, unit: Unit) -> bool:
    if spec.is_uniform:
        return spec.unit == unit
    if any(segment.unit != unit for partition in spec.partitions for segment in partition.segments):
        return False
    for partition in spec.partitions:
        if not _partition_fully_covers_axis(partition, spec.shape) and spec.unit != unit:
            return False
    return True


def _partitioned_specs_compatible_for_addition(left: ArraySpec, right: ArraySpec) -> bool:
    if not _same_partition_layout(left.partitions, right.partitions):
        return False
    for left_part, right_part in zip(left.partitions, right.partitions, strict=False):
        for left_segment, right_segment in zip(left_part.segments, right_part.segments, strict=False):
            if left_segment.unit != right_segment.unit:
                return False
        # If there are gaps in a same-layout partition, both specs fall back to
        # their uniform unit over those gaps, so the fallback units must match.
        shape = left.shape if left.shape is not None else right.shape
        if not _partition_fully_covers_axis(left_part, shape) and left.unit != right.unit:
            return False
    return True


def combine_multiplicative(
    left: ArraySpec,
    right: ArraySpec,
    op: Callable[[Unit, Unit], Unit],
) -> ArraySpec:
    """Combine specs for elementwise multiplication-like operations."""

    shape = left.shape if left.shape is not None else right.shape
    if _same_partition_layout(left.partitions, right.partitions):
        return ArraySpec(
            op(left.unit, right.unit),
            _combine_matching_partitions(left.partitions, right.partitions, op),
            shape,
        )
    if left.is_uniform:
        return ArraySpec(
            op(left.unit, right.unit),
            tuple(part.map_units(lambda unit: op(left.unit, unit)) for part in right.partitions),
            shape,
        )
    if right.is_uniform:
        return ArraySpec(
            op(left.unit, right.unit),
            tuple(part.map_units(lambda unit: op(unit, right.unit)) for part in left.partitions),
            shape,
        )
    # Both sides are partitioned differently.  Keep a conservative fallback;
    # callers can issue a diagnostic because cross-axis piecewise products are
    # representable only with a richer region model than this small library has.
    return ArraySpec(op(left.unit, right.unit), (), shape)


def _same_partition_layout(left: tuple[AxisPartition, ...], right: tuple[AxisPartition, ...]) -> bool:
    if len(left) != len(right):
        return False
    for left_part, right_part in zip(left, right, strict=False):
        if left_part.axis != right_part.axis or len(left_part.segments) != len(right_part.segments):
            return False
        for left_segment, right_segment in zip(left_part.segments, right_part.segments, strict=False):
            if (left_segment.start, left_segment.stop) != (right_segment.start, right_segment.stop):
                return False
    return True


def _combine_matching_partitions(
    left: tuple[AxisPartition, ...],
    right: tuple[AxisPartition, ...],
    op: Callable[[Unit, Unit], Unit],
) -> tuple[AxisPartition, ...]:
    partitions: list[AxisPartition] = []
    for left_part, right_part in zip(left, right, strict=False):
        segments = tuple(
            AxisSegment(
                left_segment.start,
                left_segment.stop,
                op(left_segment.unit, right_segment.unit),
            )
            for left_segment, right_segment in zip(left_part.segments, right_part.segments, strict=False)
        )
        partitions.append(AxisPartition(left_part.axis, segments))
    return tuple(partitions)


def remap_partitions_for_transpose(spec: ArraySpec, permutation: Sequence[int]) -> ArraySpec:
    if not spec.partitions:
        return spec
    axis_map = {old_axis: new_axis for new_axis, old_axis in enumerate(permutation)}
    partitions = tuple(
        partition.remap_axis(axis_map[partition.axis])
        for partition in spec.partitions
        if partition.axis in axis_map
    )
    shape = None if spec.shape is None else tuple(spec.shape[old_axis] for old_axis in permutation)
    return ArraySpec(spec.unit, partitions, shape)


def remap_partitions_for_broadcast(
    spec: ArraySpec,
    broadcast_dimensions: Sequence[int],
    output_shape: Sequence[int] | None,
) -> ArraySpec:
    if not spec.partitions:
        return spec.with_shape(output_shape)
    axis_map = {old_axis: int(new_axis) for old_axis, new_axis in enumerate(broadcast_dimensions)}
    partitions = tuple(
        partition.remap_axis(axis_map[partition.axis])
        for partition in spec.partitions
        if partition.axis in axis_map
    )
    return ArraySpec(spec.unit, partitions, output_shape)


def slice_partitions(
    spec: ArraySpec,
    starts: Sequence[int],
    limits: Sequence[int],
    strides: Sequence[int] | None,
    output_shape: Sequence[int] | None,
) -> tuple[ArraySpec, bool]:
    """Propagate partitions through static slice.

    Returns ``(new_spec, precise)``.  Non-unit strides preserve the fallback unit
    but drop partitions because intervals no longer necessarily stay contiguous.
    """

    if not spec.partitions:
        return spec.with_shape(output_shape), True
    strides = tuple(1 for _ in starts) if strides is None else tuple(strides)
    if any(stride != 1 for stride in strides):
        return ArraySpec(spec.unit, (), output_shape), False

    new_partitions: list[AxisPartition] = []
    for partition in spec.partitions:
        axis = partition.axis
        start = int(starts[axis])
        limit = int(limits[axis])
        segments: list[AxisSegment] = []
        for segment in partition.segments:
            lo = max(segment.start, start)
            hi = min(segment.stop, limit)
            if lo < hi:
                segments.append(AxisSegment(lo - start, hi - start, segment.unit))
        if segments:
            new_partitions.append(AxisPartition(axis, tuple(segments)))
    return ArraySpec(spec.unit, tuple(new_partitions), output_shape), True


def reduce_partitions(
    spec: ArraySpec,
    axes: Iterable[int],
    output_shape: Sequence[int] | None,
) -> tuple[ArraySpec, bool]:
    """Propagate partitions through reductions.

    Reducing over a partitioned axis is valid when every value included in the
    reduction has the same unit.  If the partition fully covers the reduced
    axis, the fallback unit is irrelevant; otherwise gaps use the fallback unit.
    """

    axes_set = set(int(axis) for axis in axes)
    if not spec.partitions:
        return spec.with_shape(output_shape), True
    kept: list[AxisPartition] = []
    precise = True
    reduced_common_units: list[Unit] = []
    for partition in spec.partitions:
        if partition.axis in axes_set:
            units = {segment.unit for segment in partition.segments}
            if not _partition_fully_covers_axis(partition, spec.shape):
                units.add(spec.unit)
            if len(units) == 1:
                reduced_common_units.append(next(iter(units)))
            else:
                precise = False
            continue
        removed_before = sum(1 for axis in axes_set if axis < partition.axis)
        kept.append(partition.remap_axis(partition.axis - removed_before))

    output_unit = spec.unit
    if reduced_common_units:
        unique_reduced_units = set(reduced_common_units)
        if len(unique_reduced_units) == 1:
            output_unit = next(iter(unique_reduced_units))
        else:
            precise = False
    return ArraySpec(output_unit, tuple(kept), output_shape), precise


def _partition_fully_covers_axis(partition: AxisPartition, shape: tuple[int, ...] | None) -> bool:
    if shape is None or partition.axis >= len(shape):
        return False
    expected = 0
    for segment in partition.segments:
        if segment.start != expected:
            return False
        expected = segment.stop
    return expected == int(shape[partition.axis])


def concatenate_specs(
    specs: Sequence[ArraySpec],
    axis: int,
    output_shape: Sequence[int] | None,
) -> ArraySpec:
    """Propagate units through concatenate, creating axis partitions as needed."""

    if not specs:
        return ArraySpec(ONE, (), output_shape)
    axis = int(axis)
    first = specs[0]
    if all(spec.equivalent_units(first) for spec in specs):
        return first.with_shape(output_shape)

    segments: list[AxisSegment] = []
    offset = 0
    # Heterogeneous concatenation is represented by explicit axis segments.
    # Use a neutral fallback so the whole array is not misleadingly described
    # as having the first operand's unit outside the partition metadata.
    fallback = ONE
    other_partitions: list[AxisPartition] = []
    for spec in specs:
        length = _axis_length(spec, axis)
        if spec.is_uniform:
            segments.append(AxisSegment(offset, offset + length, spec.unit))
        else:
            for partition in spec.partitions:
                if partition.axis == axis:
                    segments.extend(segment.shifted(offset) for segment in partition.segments)
                else:
                    other_partitions.append(partition)
        offset += length
    partitions: list[AxisPartition] = []
    if segments:
        partitions.append(AxisPartition(axis, tuple(segments)))
    # Preserve non-concat-axis partitions only if they are identical; otherwise
    # retaining them would imply a precision that the small model cannot encode.
    if other_partitions and len(set(other_partitions)) == 1:
        partitions.append(other_partitions[0])
    return ArraySpec(fallback, tuple(partitions), output_shape)


def _axis_length(spec: ArraySpec, axis: int) -> int:
    if spec.shape is not None and axis < len(spec.shape):
        return int(spec.shape[axis])
    lengths = [segment.stop for partition in spec.partitions if partition.axis == axis for segment in partition.segments]
    if lengths:
        return max(lengths)
    raise ValueError("cannot infer concatenate segment length without shape metadata")


def _normalize_partitions(partitions: Iterable[AxisPartition]) -> tuple[AxisPartition, ...]:
    return tuple(sorted(tuple(partitions), key=lambda partition: partition.axis))
