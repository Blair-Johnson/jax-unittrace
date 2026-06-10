"""Visual demonstration of jax_unittrace capabilities.

Run with:
    pixi run python examples/demo.py
"""

from __future__ import annotations

import textwrap

import jax
import jax.numpy as jnp
from jax import lax

from jax_unittrace import ONE, tag, trace_units, unit


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"


def line(char: str = "─") -> None:
    print(DIM + char * 88 + RESET)


def title(text: str) -> None:
    print()
    line("═")
    print(f"{BOLD}{CYAN}{text}{RESET}")
    line("═")


def section(text: str) -> None:
    print()
    print(f"{BOLD}{BLUE}▶ {text}{RESET}")
    line()


def code(src: str) -> None:
    print(DIM + textwrap.dedent(src).strip("\n") + RESET)


def status(result) -> str:
    return f"{GREEN}OK{RESET}" if result.ok else f"{RED}ERROR{RESET}"


def show_result(result, *, outputs: bool = True, equations: bool = False) -> None:
    print(f"trace status: {status(result)}")
    if outputs:
        for index, spec in enumerate(result.output_specs):
            print(f"  output[{index}] unit metadata: {MAGENTA}{spec.describe()}{RESET}  shape={spec.shape}")
    if result.diagnostics:
        print(f"  diagnostics:")
        for diagnostic in result.diagnostics:
            color = RED if diagnostic.severity == "error" else YELLOW
            print(f"    {color}{diagnostic}{RESET}")
    if equations:
        print("  equation trace:")
        for eqn in result.equation_traces:
            ins = ", ".join(spec.describe() for spec in eqn.input_specs)
            outs = ", ".join(spec.describe() for spec in eqn.output_specs)
            print(f"    #{eqn.index:02d} {eqn.primitive:18s} ({ins}) -> ({outs})")


def main() -> None:
    title("jax_unittrace: static symbolic units for ordinary JAX programs")

    m = unit("m")
    s = unit("s")
    kg = unit("kg")

    print(f"Arbitrary symbolic units: m={m}, s={s}, kg={kg}")
    print(f"Unit algebra example: acceleration = m / s**2 = {BOLD}{m / s**2}{RESET}")

    section("1. Trace an ordinary JAX program without wrapping values inside the function")
    code(
        """
        def velocity(distance, time):
            # distance and time are ordinary JAX tracers here, not custom array wrappers.
            return distance / time

        trace_units(velocity, tag(jnp.ones(4), m), tag(jnp.ones(4), s))
        """
    )

    def velocity(distance, time):
        return distance / time

    result = trace_units(velocity, tag(jnp.ones(4), m), tag(jnp.ones(4), s))
    show_result(result, equations=True)

    section("2. Catch invalid addition statically")
    code(
        """
        def bad_physics(distance, time):
            return distance + time
        """
    )

    def bad_physics(distance, time):
        return distance + time

    result = trace_units(bad_physics, tag(jnp.ones(3), m), tag(jnp.ones(3), s))
    show_result(result)

    section("3. Multiplication, powers, reductions, and matrix products do unit algebra")
    code(
        """
        def energy_like(mass, velocity):
            kinetic_scale = mass * velocity**2
            return jnp.sum(kinetic_scale)
        """
    )

    def energy_like(mass, velocity_):
        kinetic_scale = mass * velocity_**2
        return jnp.sum(kinetic_scale)

    result = trace_units(
        energy_like,
        tag(jnp.ones(5), kg),
        tag(jnp.ones(5), m / s),
    )
    show_result(result, equations=True)

    section("4. Heterogeneous concatenate creates axis partitions")
    code(
        """
        def join(distance_block, time_block):
            return jnp.concatenate([distance_block, time_block], axis=0)
        """
    )

    def join(distance_block, time_block):
        return jnp.concatenate([distance_block, time_block], axis=0)

    result = trace_units(join, tag(jnp.ones(2), m), tag(jnp.ones(3), s))
    show_result(result)
    print("  visual axis 0 partition:")
    print(f"    indices:  0       1       2       3       4")
    print(f"    units:   {GREEN}[ m ][ m ]{RESET}{YELLOW}[ s ][ s ][ s ]{RESET}")

    section("5. Partition units propagate through math")
    code(
        """
        def scale_joined(distance_block, time_block, mass):
            joined = jnp.concatenate([distance_block, time_block], axis=0)
            return joined * mass
        """
    )

    def scale_joined(distance_block, time_block, mass):
        joined = jnp.concatenate([distance_block, time_block], axis=0)
        return joined * mass

    result = trace_units(
        scale_joined,
        tag(jnp.ones(2), m),
        tag(jnp.ones(3), s),
        tag(jnp.ones(5), kg),
    )
    show_result(result)
    print("  visual axis 0 partition after multiplying by kg:")
    print(f"    units:   {GREEN}[ kg*m ][ kg*m ]{RESET}{YELLOW}[ kg*s ][ kg*s ][ kg*s ]{RESET}")

    section("6. Partition-aware checks catch mixed-unit reductions/additions")
    code(
        """
        def reduce_joined(distance_block, time_block):
            return jnp.sum(jnp.concatenate([distance_block, time_block]))
        """
    )

    def reduce_joined(distance_block, time_block):
        return jnp.sum(jnp.concatenate([distance_block, time_block]))

    result = trace_units(reduce_joined, tag(jnp.ones(2), m), tag(jnp.ones(3), s))
    show_result(result)

    print()
    print(f"{DIM}But a homogeneous partition can reduce cleanly:{RESET}")
    result = trace_units(
        lambda x: jnp.sum(x),
        tag(jnp.ones(5), axes={0: [(0, 2, m), (2, 5, m)]}),
    )
    show_result(result)

    section("7. Nested JAXPRs: jit, where, cond, and scan are interpreted too")
    code(
        """
        @jax.jit
        def choose_and_accumulate(pred, x, y):
            z = jnp.where(pred, x, y)
            return lax.cond(pred[0], lambda v: v + x, lambda v: v + y, z)
        """
    )

    @jax.jit
    def choose_and_accumulate(pred, x, y):
        z = jnp.where(pred, x, y)
        return lax.cond(pred[0], lambda v: v + x, lambda v: v + y, z)

    result = trace_units(
        choose_and_accumulate,
        jnp.array([True, False, True]),
        tag(jnp.ones(3), m),
        tag(jnp.ones(3), s),
    )
    show_result(result)

    print()
    print(f"{DIM}A valid scan example:{RESET}")
    code(
        """
        def scan_example(init, xs):
            return lax.scan(lambda carry, x: (carry + x, carry * x), init, xs)
        """
    )

    def scan_example(init, xs):
        return lax.scan(lambda carry, x: (carry + x, carry * x), init, xs)

    result = trace_units(scan_example, tag(jnp.ones(2), m), tag(jnp.ones((3, 2)), m))
    show_result(result)

    title("Demo complete")
    print(f"{GREEN}Everything above was inferred statically from JAXPRs; the user program stayed ordinary JAX.{RESET}")


if __name__ == "__main__":
    main()
