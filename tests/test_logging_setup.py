from pathlib import Path

import QuizGenerator as quizgen_pkg


def test_find_project_root_prefers_cwd_pyproject(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    resolved = quizgen_pkg._find_project_root()
    assert resolved == tmp_path.resolve()


def test_find_project_root_falls_back_to_cwd_when_no_pyproject(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(quizgen_pkg, "_find_pyproject_root", lambda _: None)

    resolved = quizgen_pkg._find_project_root()
    assert resolved == tmp_path.resolve()


def test_anchor_file_handler_paths_uses_absolute_base(tmp_path):
    config = {
        "handlers": {
            "console": {"class": "logging.StreamHandler"},
            "file": {"class": "logging.FileHandler", "filename": "out/logs/a.log"},
        }
    }
    base_dir = Path(tmp_path).resolve()

    quizgen_pkg._anchor_file_handler_paths(config, base_dir=base_dir)
    assert config["handlers"]["file"]["filename"] == str(base_dir / "out/logs/a.log")
