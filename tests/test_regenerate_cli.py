import pytest

pytest.importorskip("typer")
from typer.testing import CliRunner  # noqa: E402

import QuizGenerator.regenerate as regenerate  # noqa: E402


def test_regenerate_cli_requires_image_or_encrypted_str():
    runner = CliRunner()
    result = runner.invoke(regenerate.app, [])

    assert result.exit_code == 1
    assert "Either --image or --encrypted-str is required" in result.output


def test_regenerate_cli_rejects_image_and_encrypted_str_together():
    runner = CliRunner()
    result = runner.invoke(
        regenerate.app,
        ["--image", "scan.png", "--encrypted-str", "token"],
    )

    assert result.exit_code == 1
    assert "Cannot use both --image and --encrypted-str at the same time" in result.output


def test_regenerate_cli_encrypted_str_happy_path(monkeypatch):
    captured = {}

    def _fake_regenerate_from_encrypted(encrypted_data, points, *, image_mode, yaml_path):
        captured["call"] = {
            "encrypted_data": encrypted_data,
            "points": points,
            "image_mode": image_mode,
            "yaml_path": yaml_path,
        }
        return {
            "question_type": "fromtext",
            "seed": 123,
            "version": "1.0",
            "answers": {"kind": "essay", "data": []},
            "answer_objects": [],
            "answer_key_html": "<p>ok</p>",
            "explanation_markdown": None,
            "explanation_html": None,
        }

    monkeypatch.setattr(regenerate, "regenerate_from_encrypted", _fake_regenerate_from_encrypted)
    monkeypatch.setattr(regenerate, "display_answer_summary", lambda *_args, **_kwargs: None)

    runner = CliRunner()
    result = runner.invoke(
        regenerate.app,
        ["--encrypted-str", "abc123", "--points", "3.0", "--yaml", "replay.yaml", "--image-mode", "none"],
    )

    assert result.exit_code == 0
    assert captured["call"] == {
        "encrypted_data": "abc123",
        "points": 3.0,
        "image_mode": "none",
        "yaml_path": "replay.yaml",
    }
    assert "Successfully regenerated 1 question(s)" in result.output


def test_regenerate_cli_rejects_underscore_flag():
    runner = CliRunner()
    result = runner.invoke(regenerate.app, ["--encrypted_str", "abc123"])

    assert result.exit_code != 0
    assert "No such option: --encrypted_str" in result.output
