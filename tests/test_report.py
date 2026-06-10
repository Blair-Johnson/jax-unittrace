import jax.numpy as jnp

from jax_unittrace import format_report, tag, trace_units, unit


def test_report_api_smoke_for_error_and_success(tmp_path, capsys):
    m = unit("m")
    s = unit("s")

    bad = trace_units(lambda a, b: a + b, tag(jnp.ones(2), m), tag(jnp.ones(2), s))
    report = bad.format(equations=True)
    assert isinstance(report, str)
    assert "unit-mismatch" in report
    assert "float32" in report

    assert format_report(bad)
    bad.print_report()
    assert "unit-mismatch" in capsys.readouterr().out

    path = tmp_path / "unit-report.txt"
    bad.save_report(str(path))
    assert path.read_text()

    good = trace_units(lambda a, b: a / b, tag(jnp.ones(2), m), tag(jnp.ones(2), s))
    assert good.ok
    assert "m*s^-1" in good.format()
