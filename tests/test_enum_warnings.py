import logging

from QuizGenerator.premade_questions.cst334.memory_questions import CachingQuestion
from QuizGenerator.premade_questions.cst334.process import SchedulingQuestion


def test_invalid_scheduler_kind_warns(caplog):
    with caplog.at_level(logging.WARNING):
        kind = SchedulingQuestion.get_kind_from_string("FIDO")
    assert kind == SchedulingQuestion.Kind.FIFO
    assert any("Invalid scheduler_kind" in record.message for record in caplog.records)


def test_invalid_cache_policy_warns(caplog):
    with caplog.at_level(logging.WARNING):
        ctx = CachingQuestion._build_context(rng_seed=1, policy="FIDO")
    assert ctx["cache_policy"] == CachingQuestion.Kind.FIFO
    assert any("Invalid cache policy" in record.message for record in caplog.records)
