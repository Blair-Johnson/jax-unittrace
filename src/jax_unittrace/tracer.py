"""Static unit propagation over JAX jaxprs."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Callable, Mapping, Sequence

import jax

from .result import Diagnostic, EquationTrace, TraceResult
from .specs import (
    ArraySpec,
    compatible_for_addition,
    concatenate_specs,
    reduce_partitions,
    remap_partitions_for_broadcast,
    remap_partitions_for_transpose,
    slice_partitions,
)
from .units import ONE, UnitLike, as_unit


@dataclass(frozen=True)
class TaggedArray:
    """A value plus unit metadata consumed by :func:`trace_units`.

    Tagged values are *not* passed into the user's function.  ``trace_units``
    unwraps them before calling ``jax.make_jaxpr``, so ordinary JAX code can be
    traced without introducing custom array wrappers or custom pytrees.
    """

    value: Any
    spec: ArraySpec


def tag(
    value: Any,
    unit: UnitLike = None,
    *,
    axes: Any | None = None,
) -> TaggedArray:
    """Attach unit metadata to an input value for static tracing.

    Parameters
    ----------
    value:
        The concrete example value used by ``jax.make_jaxpr``.
    unit:
        A unit atom string, :class:`Unit`, mapping, or ``None`` for
        dimensionless.
    axes:
        Optional partition metadata.  The easiest form is a mapping from axis
        index to ``(start, stop, unit)`` intervals, e.g.
        ``{0: [(0, 3, unit("m")), (3, 5, unit("s"))]}``.
    """

    shape = getattr(value, "shape", None)
    dtype = _dtype_of_value(value)
    return TaggedArray(value, ArraySpec.from_user(unit, axes, shape, dtype))


def spec(unit: UnitLike = None, *, axes: Any | None = None) -> ArraySpec:
    """Convenience constructor for standalone input specifications."""

    return ArraySpec.from_user(unit, axes)


def trace_units(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> TraceResult:
    """Trace ``fn`` with JAX and infer units for values in its computational graph.

    Users tag only the boundary inputs.  The original function receives raw JAX
    values, and the returned :class:`TraceResult` contains output specs,
    per-equation specs, and diagnostics.
    """

    raw_args, input_specs_args = _unwrap_tree(args)
    raw_kwargs, input_specs_kwargs = _unwrap_tree(kwargs)
    raw_tree = (raw_args, raw_kwargs)
    spec_tree = (input_specs_args, input_specs_kwargs)
    raw_leaves, treedef = jax.tree_util.tree_flatten(raw_tree)
    spec_leaves, spec_treedef = jax.tree_util.tree_flatten(spec_tree)
    if spec_treedef != treedef:
        raise TypeError("internal error: raw and spec pytrees diverged")

    def wrapped(*flat_args: Any) -> Any:
        call_args, call_kwargs = jax.tree_util.tree_unflatten(treedef, flat_args)
        return fn(*call_args, **call_kwargs)

    closed_jaxpr = jax.make_jaxpr(wrapped)(*raw_leaves)
    closed = getattr(closed_jaxpr, "jaxpr", closed_jaxpr)
    invars = tuple(closed.invars)
    input_specs = tuple(
        _with_aval_metadata(input_spec, invar)
        for input_spec, invar in zip(spec_leaves, invars, strict=False)
    )
    interpreter = _Interpreter()
    output_specs, equation_traces, var_specs = interpreter.eval_closed_jaxpr(closed_jaxpr, input_specs)
    return TraceResult(
        jaxpr=closed_jaxpr,
        input_specs=input_specs,
        output_specs=output_specs,
        equation_traces=equation_traces,
        var_specs=var_specs,
        diagnostics=tuple(interpreter.diagnostics),
    )


# Friendly alias.
trace = trace_units


class _Interpreter:
    def __init__(self) -> None:
        self.diagnostics: list[Diagnostic] = []
        self._equation_counter = 0
        self._pending_nested_traces: list[EquationTrace] = []

    def eval_closed_jaxpr(
        self,
        closed_jaxpr: Any,
        input_specs: Sequence[ArraySpec],
    ) -> tuple[tuple[ArraySpec, ...], tuple[EquationTrace, ...], dict[str, ArraySpec]]:
        jaxpr = getattr(closed_jaxpr, "jaxpr", closed_jaxpr)
        consts = tuple(getattr(closed_jaxpr, "consts", ()) or ())
        return self.eval_jaxpr(jaxpr, consts, input_specs)

    def eval_jaxpr(
        self,
        jaxpr: Any,
        consts: Sequence[Any],
        input_specs: Sequence[ArraySpec],
    ) -> tuple[tuple[ArraySpec, ...], tuple[EquationTrace, ...], dict[str, ArraySpec]]:
        env: dict[Any, ArraySpec] = {}
        equation_traces: list[EquationTrace] = []

        for constvar, const in zip(tuple(getattr(jaxpr, "constvars", ())), consts, strict=False):
            env[constvar] = _dimensionless_from_aval(constvar).with_shape(getattr(const, "shape", None))
        for invar, input_spec in zip(tuple(jaxpr.invars), input_specs, strict=False):
            env[invar] = _with_aval_metadata(input_spec, invar)

        for eqn in jaxpr.eqns:
            index = self._equation_counter
            self._equation_counter += 1
            primitive = eqn.primitive.name
            in_specs = tuple(_read_env(env, invar) for invar in eqn.invars)
            pending_start = len(self._pending_nested_traces)
            out_specs = self._eval_eqn(primitive, in_specs, eqn.invars, eqn.params, eqn.outvars, index)
            nested_traces = tuple(self._pending_nested_traces[pending_start:])
            del self._pending_nested_traces[pending_start:]
            if len(out_specs) != len(eqn.outvars):
                out_specs = _fit_output_arity(out_specs, len(eqn.outvars), eqn.outvars)
            for outvar, out_spec in zip(eqn.outvars, out_specs, strict=False):
                env[outvar] = _with_aval_metadata(out_spec, outvar)
            equation_traces.append(
                EquationTrace(
                    index=index,
                    primitive=primitive,
                    input_specs=in_specs,
                    output_specs=tuple(_with_aval_metadata(spec, outvar) for spec, outvar in zip(out_specs, eqn.outvars, strict=False)),
                    params=_safe_params(eqn.params),
                    invars=tuple(str(var) for var in eqn.invars),
                    outvars=tuple(str(var) for var in eqn.outvars),
                    jaxpr=str(eqn),
                )
            )
            equation_traces.extend(nested_traces)

        outputs = tuple(_read_env(env, outvar) for outvar in jaxpr.outvars)
        var_specs = {str(var): spec for var, spec in env.items()}
        return outputs, tuple(equation_traces), var_specs

    def _eval_eqn(
        self,
        primitive: str,
        inputs: tuple[ArraySpec, ...],
        invars: Sequence[Any],
        params: Mapping[str, Any],
        outvars: Sequence[Any],
        equation_index: int,
    ) -> tuple[ArraySpec, ...]:
        output_shape = _shape_of_var(outvars[0]) if outvars else None

        if primitive in {"add", "sub"}:
            self._check_additive(inputs[0], inputs[1], primitive, equation_index)
            return (_merge_additive_output(inputs[0], inputs[1]).with_shape(output_shape),)

        if primitive in {"mul", "atan2"}:
            if primitive == "atan2":
                self._require_dimensionless(inputs[0], primitive, equation_index)
                self._require_dimensionless(inputs[1], primitive, equation_index)
                return (ArraySpec(ONE, (), output_shape),)
            self._warn_if_cross_partition_product(inputs, primitive, equation_index)
            return (inputs[0].multiply(inputs[1]).with_shape(output_shape),)

        if primitive in {"div", "rem"}:
            self._warn_if_cross_partition_product(inputs, primitive, equation_index)
            return (inputs[0].divide(inputs[1]).with_shape(output_shape),)

        if primitive == "integer_pow":
            return (inputs[0].power(int(params["y"])).with_shape(output_shape),)

        if primitive == "square":
            return (inputs[0].power(2).with_shape(output_shape),)

        if primitive == "pow":
            self._require_dimensionless(inputs[1], primitive, equation_index, role="exponent")
            exponent = _literal_value(params.get("y"))
            if exponent is None and len(invars) > 1:
                exponent = _literal_value(getattr(invars[1], "val", None))
            if exponent is None:
                self._warn(
                    "nonliteral-power",
                    "power with a traced exponent preserves the base unit conservatively",
                    primitive,
                    equation_index,
                )
                return (inputs[0].with_shape(output_shape),)
            return (inputs[0].power(Fraction(exponent)).with_shape(output_shape),)

        if primitive in {"neg", "abs", "copy", "copy_to_host_async", "stop_gradient", "real"}:
            return (inputs[0].with_shape(output_shape),)

        if primitive == "conj":
            return (inputs[0].with_shape(output_shape),)

        if primitive in {"convert_element_type", "bitcast_convert_type"}:
            return (inputs[0].with_shape(output_shape),)

        if primitive in {"sqrt"}:
            return (inputs[0].power(Fraction(1, 2)).with_shape(output_shape),)

        if primitive in {"rsqrt"}:
            return (inputs[0].power(Fraction(-1, 2)).with_shape(output_shape),)

        if primitive in _DIMENSIONLESS_INPUT_OUTPUT_PRIMITIVES:
            for input_spec in inputs:
                self._require_dimensionless(input_spec, primitive, equation_index)
            return (ArraySpec(ONE, (), output_shape),)

        if primitive in _ELEMENTWISE_EXTREMUM_PRIMITIVES:
            self._check_additive(inputs[0], inputs[1], primitive, equation_index)
            return (_merge_additive_output(inputs[0], inputs[1]).with_shape(output_shape),)

        if primitive in _COMPARISON_PRIMITIVES:
            self._check_additive(inputs[0], inputs[1], primitive, equation_index)
            return (ArraySpec(ONE, (), output_shape),)

        if primitive in {"broadcast_in_dim"}:
            dims = params.get("broadcast_dimensions", ())
            return (remap_partitions_for_broadcast(inputs[0], dims, output_shape),)

        if primitive in {"transpose"}:
            permutation = params.get("permutation") or params.get("dimensions")
            return (remap_partitions_for_transpose(inputs[0], permutation).with_shape(output_shape),)

        if primitive in {"reshape", "squeeze", "rev", "pad"}:
            if inputs[0].partitions and primitive in {"reshape", "squeeze"}:
                self._warn(
                    "partition-dropped",
                    f"{primitive} preserved the uniform unit but dropped axis partitions",
                    primitive,
                    equation_index,
                )
                return (inputs[0].without_partitions().with_shape(output_shape),)
            return (inputs[0].with_shape(output_shape),)

        if primitive == "slice":
            starts = params.get("start_indices", ())
            limits = params.get("limit_indices", ())
            strides = params.get("strides")
            sliced, precise = slice_partitions(inputs[0], starts, limits, strides, output_shape)
            if not precise:
                self._warn(
                    "partition-dropped",
                    "slice with non-unit strides dropped axis partitions",
                    primitive,
                    equation_index,
                )
            return (sliced,)

        if primitive in {"dynamic_slice", "gather"}:
            if inputs and inputs[0].partitions:
                self._warn(
                    "partition-dropped",
                    f"{primitive} preserved the uniform unit but dropped axis partitions",
                    primitive,
                    equation_index,
                )
            return (inputs[0].without_partitions().with_shape(output_shape),)

        if primitive in {"concatenate"}:
            dimension = params.get("dimension", params.get("axis", 0))
            return (concatenate_specs(inputs, dimension, output_shape),)

        if primitive in _ADDITIVE_REDUCTIONS:
            axes = params.get("axes", params.get("dimensions", ()))
            reduced, precise = reduce_partitions(inputs[0], axes, output_shape)
            if not precise:
                self._error(
                    "partitioned-reduction-mismatch",
                    "reduction over a partitioned axis would add values with different units",
                    primitive,
                    equation_index,
                    input=inputs[0].describe(),
                    axes=tuple(axes),
                )
            return (reduced,)

        if primitive in _EXTREMUM_REDUCTIONS:
            axes = params.get("axes", params.get("dimensions", ()))
            reduced, precise = reduce_partitions(inputs[0], axes, output_shape)
            if not precise:
                self._warn(
                    "partition-dropped",
                    f"{primitive} over a partitioned axis dropped axis partitions",
                    primitive,
                    equation_index,
                )
            return (reduced,)

        if primitive in _MULTIPLICATIVE_REDUCTIONS:
            if inputs[0].partitions:
                self._warn(
                    "partition-dropped",
                    f"{primitive} preserved the fallback unit but dropped partitions",
                    primitive,
                    equation_index,
                )
            exponent = _reduction_size(inputs[0], params)
            unit_spec = inputs[0].without_partitions()
            if exponent is not None:
                unit_spec = unit_spec.power(exponent)
            return (unit_spec.with_shape(output_shape),)

        if primitive in {"dot_general"}:
            self._warn_if_cross_partition_product(inputs[:2], primitive, equation_index)
            result = inputs[0].multiply(inputs[1]).without_partitions().with_shape(output_shape)
            if inputs[0].partitions or inputs[1].partitions:
                self._warn(
                    "partition-dropped",
                    "dot_general multiplied units but dropped axis partitions",
                    primitive,
                    equation_index,
                )
            return (result,)

        if primitive in {"select_n"}:
            # JAX encodes where/select as predicate followed by branch values.
            branch_specs = inputs[1:]
            if branch_specs:
                first = branch_specs[0]
                for branch in branch_specs[1:]:
                    self._check_additive(first, branch, primitive, equation_index)
                return (_merge_many_additive_outputs(branch_specs).with_shape(output_shape),)
            return (ArraySpec(ONE, (), output_shape),)

        if primitive in {"iota", "rng_bit_generator", "random_seed", "random_split"}:
            return tuple(_dimensionless_from_aval(outvar) for outvar in outvars)


        if primitive == "cond":
            return self._eval_cond(inputs, params, outvars, equation_index)

        if primitive == "scan":
            return self._eval_scan(inputs, params, outvars, equation_index)
        if "jaxpr" in params and primitive in {"jit", "pjit", "xla_call", "call"}:
            nested = params["jaxpr"]
            nested_outputs, nested_traces, _ = self.eval_closed_jaxpr(nested, inputs)
            self._pending_nested_traces.extend(nested_traces)
            return tuple(spec.with_shape(_shape_of_var(outvar)) for spec, outvar in zip(nested_outputs, outvars, strict=False))

        return self._fallback(primitive, inputs, outvars, equation_index)

    def _eval_cond(
        self,
        inputs: tuple[ArraySpec, ...],
        params: Mapping[str, Any],
        outvars: Sequence[Any],
        equation_index: int,
    ) -> tuple[ArraySpec, ...]:
        # JAX cond inputs are selector/index followed by shared branch operands.
        branch_inputs = inputs[1:]
        branches = tuple(params.get("branches", ()))
        if not branches:
            return tuple(_dimensionless_from_aval(outvar) for outvar in outvars)

        branch_outputs: list[tuple[ArraySpec, ...]] = []
        for branch in branches:
            outputs, traces, _ = self.eval_closed_jaxpr(branch, branch_inputs)
            branch_outputs.append(outputs)
            self._pending_nested_traces.extend(traces)
        merged: list[ArraySpec] = []
        for output_index, output_group in enumerate(zip(*branch_outputs, strict=False)):
            first = output_group[0]
            for branch_spec in output_group[1:]:
                if not compatible_for_addition(first, branch_spec):
                    self._error(
                        "unit-mismatch",
                        f"cond branch {output_index} units differ: {first.describe()} vs {branch_spec.describe()}",
                        "cond",
                        equation_index,
                        left=first.describe(),
                        right=branch_spec.describe(),
                    )
            merged.append(_merge_many_additive_outputs(output_group))
        return tuple(spec.with_shape(_shape_of_var(outvar)) for spec, outvar in zip(merged, outvars, strict=False))

    def _eval_scan(
        self,
        inputs: tuple[ArraySpec, ...],
        params: Mapping[str, Any],
        outvars: Sequence[Any],
        equation_index: int,
    ) -> tuple[ArraySpec, ...]:
        body_jaxpr = params.get("jaxpr")
        if body_jaxpr is None:
            return self._fallback("scan", inputs, outvars, equation_index)

        num_consts = int(params.get("num_consts", 0))
        num_carry = int(params.get("num_carry", 0))
        const_specs = inputs[:num_consts]
        carry_specs = inputs[num_consts : num_consts + num_carry]
        xs_specs = inputs[num_consts + num_carry :]
        body_inputs = tuple(const_specs) + tuple(carry_specs) + tuple(
            self._scan_slice_input_spec(spec, equation_index) for spec in xs_specs
        )
        body_outputs, body_traces, _ = self.eval_closed_jaxpr(body_jaxpr, body_inputs)
        self._pending_nested_traces.extend(body_traces)

        carry_outputs = body_outputs[:num_carry]
        for carry_input, carry_output in zip(carry_specs, carry_outputs, strict=False):
            if not compatible_for_addition(carry_input, carry_output):
                self._error(
                    "unit-mismatch",
                    f"scan carry unit changed from {carry_input.describe()} to {carry_output.describe()}",
                    "scan",
                    equation_index,
                    left=carry_input.describe(),
                    right=carry_output.describe(),
                )

        result_specs: list[ArraySpec] = []
        for output_index, output_spec in enumerate(body_outputs):
            outvar = outvars[output_index] if output_index < len(outvars) else None
            shaped = output_spec.with_shape(_shape_of_var(outvar))
            if output_index >= num_carry:
                shaped = _stack_scan_axis(shaped, _shape_of_var(outvar))
            result_specs.append(shaped)
        return tuple(result_specs)

    def _scan_slice_input_spec(self, spec: ArraySpec, equation_index: int) -> ArraySpec:
        if spec.partitions:
            self._warn(
                "partition-dropped",
                "scan over a partitioned input preserved the fallback unit but dropped partitions",
                "scan",
                equation_index,
            )
            return spec.without_partitions()
        return spec

    def _fallback(
        self,
        primitive: str,
        inputs: tuple[ArraySpec, ...],
        outvars: Sequence[Any],
        equation_index: int,
    ) -> tuple[ArraySpec, ...]:
        if not inputs:
            return tuple(_dimensionless_from_aval(outvar) for outvar in outvars)
        if len(inputs) == 1:
            return tuple(inputs[0].with_shape(_shape_of_var(outvar)) for outvar in outvars)
        first = inputs[0]
        compatible = all(compatible_for_addition(first, other) for other in inputs[1:])
        if compatible:
            result = first
        else:
            self._warn(
                "unsupported-primitive",
                f"unsupported primitive {primitive!r}; output unit set to dimensionless",
                primitive,
                equation_index,
            )
            result = ArraySpec(ONE)
        return tuple(result.with_shape(_shape_of_var(outvar)) for outvar in outvars)

    def _check_additive(
        self,
        left: ArraySpec,
        right: ArraySpec,
        primitive: str,
        equation_index: int,
    ) -> None:
        if not compatible_for_addition(left, right):
            self._error(
                "unit-mismatch",
                _unit_mismatch_message(primitive, left, right),
                primitive,
                equation_index,
                left=left.describe(),
                right=right.describe(),
            )

    def _require_dimensionless(
        self,
        input_spec: ArraySpec,
        primitive: str,
        equation_index: int,
        *,
        role: str = "input",
    ) -> None:
        if any(unit != ONE for unit in input_spec.all_units()):
            self._error(
                "expected-dimensionless",
                f"{primitive} requires a dimensionless {role}, got {input_spec.describe()}",
                primitive,
                equation_index,
                input=input_spec.describe(),
            )

    def _warn_if_cross_partition_product(
        self,
        inputs: Sequence[ArraySpec],
        primitive: str,
        equation_index: int,
    ) -> None:
        if len(inputs) >= 2 and inputs[0].partitions and inputs[1].partitions and inputs[0].partitions != inputs[1].partitions:
            self._warn(
                "partition-precision-loss",
                f"{primitive} combines different partition layouts; output keeps only a fallback unit",
                primitive,
                equation_index,
            )

    def _error(self, code: str, message: str, primitive: str, equation_index: int, **details: Any) -> None:
        self.diagnostics.append(Diagnostic("error", code, message, primitive, equation_index, details))

    def _warn(self, code: str, message: str, primitive: str, equation_index: int, **details: Any) -> None:
        self.diagnostics.append(Diagnostic("warning", code, message, primitive, equation_index, details))


def _unit_mismatch_message(primitive: str, left: ArraySpec, right: ArraySpec) -> str:
    if primitive in {"add", "sub"}:
        action = "add" if primitive == "add" else "subtract"
        return f"cannot {action} values with units {left.describe()} and {right.describe()}"
    if primitive in {"select_n", "cond"}:
        return f"branch alternatives have incompatible units {left.describe()} and {right.describe()}"
    if primitive in {"eq", "ne", "ge", "gt", "le", "lt", "max", "min"}:
        return f"cannot compare ordered values with units {left.describe()} and {right.describe()}"
    return f"unit mismatch in {primitive}: expected compatible units, got {left.describe()} and {right.describe()}"

_DIMENSIONLESS_INPUT_OUTPUT_PRIMITIVES = {
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "cos",
    "cosh",
    "erf",
    "erfc",
    "exp",
    "exp2",
    "expm1",
    "log",
    "log1p",
    "logistic",
    "sin",
    "sinh",
    "tan",
    "tanh",
}

_COMPARISON_PRIMITIVES = {"eq", "ne", "ge", "gt", "le", "lt"}
_ELEMENTWISE_EXTREMUM_PRIMITIVES = {"max", "min"}
_ADDITIVE_REDUCTIONS = {"reduce_sum", "reduce_window_sum", "cumsum"}
_MULTIPLICATIVE_REDUCTIONS = {"reduce_prod", "cumprod"}
_EXTREMUM_REDUCTIONS = {"reduce_max", "reduce_min"}


def _stack_scan_axis(spec: ArraySpec, shape: tuple[int, ...] | None) -> ArraySpec:
    if not spec.partitions:
        return spec.with_shape(shape)
    return ArraySpec(
        spec.unit,
        tuple(partition.remap_axis(partition.axis + 1) for partition in spec.partitions),
        shape,
    )


def _unwrap_tree(value: Any) -> tuple[Any, Any]:
    if isinstance(value, TaggedArray):
        return value.value, value.spec
    if isinstance(value, tuple):
        raw_items, spec_items = zip(*(_unwrap_tree(item) for item in value), strict=False) if value else ((), ())
        return tuple(raw_items), tuple(spec_items)
    if isinstance(value, list):
        raw_items, spec_items = zip(*(_unwrap_tree(item) for item in value), strict=False) if value else ((), ())
        return list(raw_items), list(spec_items)
    if isinstance(value, dict):
        raw: dict[Any, Any] = {}
        specs: dict[Any, Any] = {}
        for key, item in value.items():
            raw_item, spec_item = _unwrap_tree(item)
            raw[key] = raw_item
            specs[key] = spec_item
        return raw, specs
    return value, _default_spec_for_value(value)


def _default_spec_for_value(value: Any) -> ArraySpec:
    return ArraySpec(ONE, (), getattr(value, "shape", None), _dtype_of_value(value))

def _dtype_of_value(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    return str(dtype)


def _dimensionless_from_aval(var: Any) -> ArraySpec:
    return ArraySpec(ONE, (), _shape_of_var(var), _dtype_of_var(var))


def _with_aval_metadata(spec: ArraySpec, var: Any) -> ArraySpec:
    shape = _shape_of_var(var)
    dtype = _dtype_of_var(var)
    if shape is None and dtype is None:
        return spec
    return spec.with_metadata(shape if shape is not None else spec.shape, dtype if dtype is not None else spec.dtype)


def _shape_of_var(var: Any) -> tuple[int, ...] | None:
    aval = getattr(var, "aval", None)
    shape = getattr(aval, "shape", None)
    if shape is None:
        value = getattr(var, "val", None)
        shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(x) for x in shape)
    except TypeError:
        return None




def _dtype_of_var(var: Any) -> str | None:
    aval = getattr(var, "aval", None)
    dtype = getattr(aval, "dtype", None)
    if dtype is None:
        value = getattr(var, "val", None)
        dtype = getattr(value, "dtype", None)
    if dtype is None:
        return None
    return str(dtype)
def _read_env(env: Mapping[Any, ArraySpec], var: Any) -> ArraySpec:
    value = getattr(var, "val", None)
    if value is not None:
        return ArraySpec(ONE, (), getattr(value, "shape", _shape_of_var(var)), _dtype_of_value(value))
    try:
        found = var in env
    except TypeError:
        found = False
    if found:
        return env[var]
    return ArraySpec(ONE, (), _shape_of_var(var), _dtype_of_var(var))


def _merge_additive_output(left: ArraySpec, right: ArraySpec) -> ArraySpec:
    if left.equivalent_units(right):
        return left
    if right.is_uniform and compatible_for_addition(left, right):
        if left.is_uniform:
            return left
        return ArraySpec(right.unit, left.partitions, left.shape)
    if left.is_uniform and compatible_for_addition(left, right):
        if right.is_uniform:
            return right
        return ArraySpec(left.unit, right.partitions, right.shape)
    return left


def _merge_many_additive_outputs(specs: Sequence[ArraySpec]) -> ArraySpec:
    result = specs[0]
    for item in specs[1:]:
        result = _merge_additive_output(result, item)
    return result


def _literal_value(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive for unusual tracers
            return None
    return None


def _fit_output_arity(specs: Sequence[ArraySpec], arity: int, outvars: Sequence[Any]) -> tuple[ArraySpec, ...]:
    if arity == 0:
        return ()
    if not specs:
        return tuple(_dimensionless_from_aval(outvar) for outvar in outvars)
    if len(specs) == arity:
        return tuple(specs)
    if len(specs) == 1:
        return tuple(specs[0].with_shape(_shape_of_var(outvar)) for outvar in outvars)
    return tuple(specs[:arity])


def _reduction_size(input_spec: ArraySpec, params: Mapping[str, Any]) -> int | None:
    axes = params.get("axes", params.get("dimensions", ()))
    if input_spec.shape is None or not axes:
        return None
    size = 1
    for axis in axes:
        size *= int(input_spec.shape[int(axis)])
    return size


def _safe_params(params: Mapping[str, Any]) -> Mapping[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in params.items():
        if key == "jaxpr":
            safe[key] = "<jaxpr>"
        else:
            safe[key] = value
    return safe
