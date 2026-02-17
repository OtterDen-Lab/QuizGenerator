import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*args: str):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-m", "QuizGenerator.generate", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def test_cli_top_level_help_lists_subcommands():
    result = _run_cli("--help")
    assert result.returncode == 0
    for subcommand in ("generate", "practice", "test", "deps", "tags"):
        assert subcommand in result.stdout


def test_cli_subcommand_help_smoke():
    for subcommand in ("generate", "practice", "test", "deps", "tags"):
        result = _run_cli(subcommand, "--help")
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()


def test_cli_generate_safe_yaml_without_pdf_succeeds():
    result = _run_cli(
        "generate",
        "--yaml",
        "example_files/example_exam_safe.yaml",
        "--num_pdfs",
        "0",
    )
    assert result.returncode == 0, result.stderr


def test_cli_tags_list_succeeds():
    result = _run_cli("tags", "list")
    assert result.returncode == 0, result.stderr
    assert "Analyzed question types" in result.stdout
