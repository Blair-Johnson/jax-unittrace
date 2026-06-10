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


class UnitTraceError(ValueError):
    """Raised by :meth:`TraceResult.raise_on_error` when diagnostics exist."""

    def __init__(self, message: str, diagnostics: Sequence[Diagnostic]):
        super().__init__(message)
        self.diagnostics = tuple(diagnostics)
