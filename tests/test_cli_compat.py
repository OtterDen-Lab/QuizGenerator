from __future__ import annotations

import sys

import QuizGenerator.generate as generate_module


def test_generate_py_compat_rewrites_nested_command(monkeypatch):
    captured = {}

    def _fake_typer_main():
        captured["argv"] = list(sys.argv)

    monkeypatch.setattr("QuizGenerator.typer_cli.main", _fake_typer_main)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "QuizGenerator/generate.py",
            "generate",
            "import-canvas",
            "--course-id",
            "31580",
        ],
    )

    generate_module.main()

    assert captured["argv"] == [
        "QuizGenerator/generate.py",
        "import-canvas",
        "--course-id",
        "31580",
    ]
