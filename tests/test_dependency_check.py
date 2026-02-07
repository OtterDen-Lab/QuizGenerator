from unittest.mock import patch

import pytest

from QuizGenerator.generate import _check_dependencies


def test_check_deps_missing_typst():
    with patch("QuizGenerator.generate.shutil.which") as which:
        which.side_effect = lambda name: None if name == "typst" else "/usr/bin/ok"
        ok, missing = _check_dependencies(require_typst=True, require_latex=False)
        assert not ok
        assert any("Typst not found" in msg for msg in missing)


def test_check_deps_missing_latexmk():
    with patch("QuizGenerator.generate.shutil.which") as which:
        which.side_effect = lambda name: None if name == "latexmk" else "/usr/bin/ok"
        ok, missing = _check_dependencies(require_typst=False, require_latex=True)
        assert not ok
        assert any("latexmk not found" in msg for msg in missing)


def test_check_deps_warns_on_pandoc(caplog):
    with patch("QuizGenerator.generate.shutil.which") as which:
        which.side_effect = lambda name: None if name == "pandoc" else "/usr/bin/ok"
        with patch("QuizGenerator.generate.log.warning") as warn:
            ok, missing = _check_dependencies(require_typst=False, require_latex=False)
        assert ok
        assert missing == []
        warn.assert_called()
        assert any("Pandoc not found" in str(call.args[0]) for call in warn.call_args_list)
