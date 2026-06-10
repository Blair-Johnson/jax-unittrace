"""Trace result and diagnostic containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .specs import ArraySpec


@dataclass(frozen=True)
class Diagnostic:
    """A unit-tracing diagnostic produced while interpreting a JAX graph."""

    severity: str
    code: str
    message: str
    primitive: str | None = None
    equation_index: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        location = ""
        if self.primitive is not None:
            location += f" in {self.primitive}"
        if self.equation_index is not None:
            location += f" at equation {self.equation_index}"
        return f"[{self.severity}] {self.code}{location}: {self.message}"


@dataclass(frozen=True)
class EquationTrace:
    """Unit metadata inferred for one JAX equation."""

    index: int
    primitive: str
    input_specs: tuple[ArraySpec, ...]
    output_specs: tuple[ArraySpec, ...]
    params: Mapping[str, Any] = field(default_factory=dict)
    invars: tuple[str, ...] = ()
    outvars: tuple[str, ...] = ()
    jaxpr: str = ""


@dataclass(frozen=True)
class TraceResult:
    """Result returned by :func:`jax_unittrace.trace_units`."""

    jaxpr: Any
    input_specs: tuple[ArraySpec, ...]
    output_specs: tuple[ArraySpec, ...]
    equation_traces: tuple[EquationTrace, ...]
    var_specs: Mapping[str, ArraySpec]
    diagnostics: tuple[Diagnostic, ...]

    @property
    def errors(self) -> tuple[Diagnostic, ...]:
        return tuple(d for d in self.diagnostics if d.severity == "error")

    @property
    def warnings(self) -> tuple[Diagnostic, ...]:
        return tuple(d for d in self.diagnostics if d.severity == "warning")

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_on_error(self) -> "TraceResult":
        if self.errors:
            joined = "\n".join(str(error) for error in self.errors)
            raise UnitTraceError(joined, self.errors)
        return self

    def format(
        self,
        *,
        equations: bool = False,
        color: bool = False,
        max_equations: int | None = 25,
        diagnostic_context: bool = True,
        context: int = 2,
    ) -> str:
        """Return a user-friendly text report for this trace result.

        Parameters
        ----------
        equations:
            Include a primitive-by-primitive unit propagation table.
        color:
            Emit ANSI colors for terminal display.
        max_equations:
            Maximum number of equations to show when ``equations=True``.  Use
            ``None`` to show every equation.
        diagnostic_context:
            Include local equation windows around diagnostics.
        context:
            Number of neighboring equations to show before/after each diagnostic.
        """

        return format_report(
            self,
            equations=equations,
            color=color,
            max_equations=max_equations,
            diagnostic_context=diagnostic_context,
            context=context,
        )

    def print_report(self, **kwargs: Any) -> None:
        """Print :meth:`format` to stdout.

        Keyword arguments are forwarded to :meth:`format`.  This is convenient
        in notebooks and debugging scripts::

            trace_units(fn, ...).print_report(equations=True, color=True)
        """

        print(self.format(**kwargs))

    def save_report(self, path: str, **kwargs: Any) -> None:
        """Write :meth:`format` output to a text file."""

        with open(path, "w", encoding="utf-8") as file:
            file.write(self.format(**kwargs))
            file.write("\n")

    def __str__(self) -> str:
        return self.format()


class UnitTraceError(ValueError):
    """Raised by :meth:`TraceResult.raise_on_error` when diagnostics exist."""

    def __init__(self, message: str, diagnostics: Sequence[Diagnostic]):
        super().__init__(message)
        self.diagnostics = tuple(diagnostics)


def format_report(
    result: TraceResult,
    *,
    equations: bool = False,
    color: bool = False,
    max_equations: int | None = 25,
    diagnostic_context: bool = True,
    context: int = 2,
) -> str:
    """Format a :class:`TraceResult` as an interpretable text report."""

    palette = _Palette(color)
    lines: list[str] = []
    lines.append(palette.heading("jax_unittrace report"))
    lines.append(_rule())
    lines.append(f"status: {palette.ok('OK') if result.ok else palette.error('ERROR')}")
    lines.append(
        f"summary: {len(result.input_specs)} input(s), {len(result.output_specs)} output(s), "
        f"{len(result.equation_traces)} equation(s), {len(result.errors)} error(s), "
        f"{len(result.warnings)} warning(s)"
    )

    lines.append("")
    lines.append(palette.section("Inputs"))
    for index, spec in enumerate(result.input_specs):
        lines.extend(_format_spec_row(f"input[{index}]", spec))

    lines.append("")
    lines.append(palette.section("Outputs"))
    for index, spec in enumerate(result.output_specs):
        lines.extend(_format_spec_row(f"output[{index}]", spec))

    if result.diagnostics:
        lines.append("")
        lines.append(palette.section("Diagnostics"))
        for diagnostic in result.diagnostics:
            paint = palette.error if diagnostic.severity == "error" else palette.warning
            lines.append(paint(f"- {diagnostic}"))
            if diagnostic.details:
                detail_text = ", ".join(f"{key}={value}" for key, value in diagnostic.details.items())
                lines.append(f"    details: {detail_text}")
    else:
        lines.append("")
        lines.append(palette.section("Diagnostics"))
        lines.append(palette.ok("- none"))

    if result.diagnostics and diagnostic_context:
        lines.append("")
        lines.append(palette.section("Debug context"))
        lines.extend(_format_diagnostic_context(result, palette, context=context))

    if equations:
        lines.append("")
        lines.append(palette.section("Equation trace"))
        shown = result.equation_traces if max_equations is None else result.equation_traces[:max_equations]
        if not shown:
            lines.append("- no equations")
        else:
            for equation in shown:
                lines.append(f"  {_format_equation(equation)}")
            remaining = len(result.equation_traces) - len(shown)
            if remaining > 0:
                lines.append(f"  ... {remaining} more equation(s); pass max_equations=None to show all")

    return "\n".join(lines)


def _format_spec_row(label: str, spec: ArraySpec) -> list[str]:
    shape = "unknown" if spec.shape is None else spec.shape
    dtype = "unknown" if spec.dtype is None else spec.dtype
    lines = [f"- {label}: unit={spec.describe()}  shape={shape}  dtype={dtype}"]
    for partition in spec.partitions:
        segments = " | ".join(
            f"[{segment.start}:{segment.stop}) {segment.unit}" for segment in partition.segments
        )
        lines.append(f"    partitioned axis {partition.axis}: {segments}")
        if spec.shape is not None and partition.axis < len(spec.shape):
            lines.append(f"      axis length: {spec.shape[partition.axis]}, fallback unit outside segments: {spec.unit}")
    return lines


def _format_diagnostic_context(
    result: TraceResult,
    palette: "_Palette",
    *,
    context: int,
) -> list[str]:
    lines: list[str] = []
    by_index = {equation.index: position for position, equation in enumerate(result.equation_traces)}
    seen: set[int] = set()
    for diagnostic in result.diagnostics:
        if diagnostic.equation_index is None or diagnostic.equation_index not in by_index:
            continue
        center = by_index[diagnostic.equation_index]
        if diagnostic.equation_index in seen:
            continue
        seen.add(diagnostic.equation_index)
        start = max(0, center - max(context, 0))
        stop = min(len(result.equation_traces), center + max(context, 0) + 1)
        lines.append(f"- around equation #{diagnostic.equation_index} ({diagnostic.primitive}):")
        for equation in result.equation_traces[start:stop]:
            is_focus = equation.index == diagnostic.equation_index
            marker = "=>" if is_focus else "  "
            rendered = _format_equation(equation)
            if is_focus:
                rendered = palette.error(rendered)
            lines.append(f"    {marker} {rendered}")
            if is_focus and equation.jaxpr:
                lines.append(f"       jaxpr: {equation.jaxpr}")
    if not lines:
        lines.append("- no equation context available")
    return lines


def _format_equation(equation: EquationTrace) -> str:
    inputs = ", ".join(
        _format_var(f"in{index}", spec) for index, spec in enumerate(equation.input_specs)
    ) or "∅"
    outputs = ", ".join(
        _format_var(f"out{index}", spec) for index, spec in enumerate(equation.output_specs)
    ) or "∅"
    return f"#{equation.index:02d} {outputs} = {equation.primitive}[units] {inputs}"


def _format_var(name: str, spec: ArraySpec) -> str:
    dtype = "?" if spec.dtype is None else spec.dtype
    shape = "?" if spec.shape is None else "[" + ",".join(str(dim) for dim in spec.shape) + "]"
    return f"{name}:{dtype}{shape}{{{spec.describe()}}}"


def _rule() -> str:
    return "-" * 80

class _Palette:
    def __init__(self, color: bool):
        self.color = color

    def heading(self, text: str) -> str:
        return self._paint(text, "\033[1;36m")

    def section(self, text: str) -> str:
        return self._paint(text, "\033[1;34m")

    def ok(self, text: str) -> str:
        return self._paint(text, "\033[32m")

    def warning(self, text: str) -> str:
        return self._paint(text, "\033[33m")

    def error(self, text: str) -> str:
        return self._paint(text, "\033[31m")

    def _paint(self, text: str, code: str) -> str:
        if not self.color:
            return text
        return f"{code}{text}\033[0m"
