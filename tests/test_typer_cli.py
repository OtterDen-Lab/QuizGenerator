import pytest

pytest.importorskip("typer")
from typer.testing import CliRunner  # noqa: E402

import QuizGenerator.typer_cli as typer_cli  # noqa: E402


def test_typer_generate_command_invokes_core_functions(monkeypatch):
    captured = {}

    def _fake_configure(**kwargs):
        captured["runtime"] = kwargs

    def _fake_deps(**kwargs):
        captured["deps"] = kwargs

    def _fake_clear():
        captured["cleared"] = True

    def _fake_generate_quiz(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(typer_cli, "_configure_runtime", _fake_configure)
    monkeypatch.setattr(typer_cli, "_ensure_dependencies", _fake_deps)
    monkeypatch.setattr(typer_cli.PerformanceTracker, "clear_metrics", _fake_clear)
    monkeypatch.setattr(typer_cli, "generate_quiz", _fake_generate_quiz)

    runner = CliRunner()
    result = runner.invoke(typer_cli.app, ["generate", "--yaml", "quiz.yaml", "--num-pdfs", "2"])

    assert result.exit_code == 0
    assert captured["args"] == ("quiz.yaml",)
    assert captured["deps"] == {"use_typst": True}
    assert captured["cleared"] is True
    assert captured["kwargs"]["num_pdfs"] == 2
    assert captured["kwargs"]["consistent_pages"] is True
    assert captured["kwargs"]["use_typst"] is True


def test_typer_handles_quizgen_error(monkeypatch):
    monkeypatch.setattr(typer_cli, "_configure_runtime", lambda **kwargs: None)

    def _raise_error(**kwargs):
        raise typer_cli.QuizGenError("synthetic failure")

    monkeypatch.setattr(typer_cli, "_ensure_dependencies", _raise_error)

    runner = CliRunner()
    result = runner.invoke(typer_cli.app, ["deps"])

    assert result.exit_code == 1
    assert "synthetic failure" in result.output


def test_typer_rejects_underscore_flag_names():
    runner = CliRunner()
    result = runner.invoke(
        typer_cli.app,
        ["generate", "--yaml", "quiz.yaml", "--num_pdfs", "1"],
    )

    assert result.exit_code != 0
    assert "No such option: --num_pdfs" in result.output
