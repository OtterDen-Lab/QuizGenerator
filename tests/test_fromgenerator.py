import pytest


def test_fromgenerator_disabled_by_default(monkeypatch):
    monkeypatch.delenv("QUIZGEN_ALLOW_GENERATOR", raising=False)
    from QuizGenerator.premade_questions.basic import FromGenerator

    with pytest.raises(ValueError, match="FromGenerator is disabled by default"):
        FromGenerator(
            name="Gen",
            points_value=1.0,
            generator="return 'hi'",
        )


def test_fromgenerator_enabled_with_env(monkeypatch):
    monkeypatch.setenv("QUIZGEN_ALLOW_GENERATOR", "1")
    from QuizGenerator.premade_questions.basic import FromGenerator

    q = FromGenerator(
        name="Gen",
        points_value=1.0,
        generator="return 'ok'",
    )
    instance = q.instantiate(rng_seed=123)
    assert instance.body is not None
