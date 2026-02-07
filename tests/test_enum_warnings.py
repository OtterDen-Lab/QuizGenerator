from unittest.mock import patch

from QuizGenerator.premade_questions.cst334.memory_questions import CachingQuestion
from QuizGenerator.premade_questions.cst334.process import SchedulingQuestion


def test_invalid_scheduler_kind_warns():
    with patch("QuizGenerator.premade_questions.cst334.process.log.warning") as warn:
        kind = SchedulingQuestion.get_kind_from_string("FIDO")
    assert kind == SchedulingQuestion.Kind.FIFO
    warn.assert_called()
    assert any("Invalid scheduler_kind" in str(call.args[0]) for call in warn.call_args_list)


def test_invalid_cache_policy_warns():
    with patch("QuizGenerator.premade_questions.cst334.memory_questions.log.warning") as warn:
        ctx = CachingQuestion._build_context(rng_seed=1, policy="FIDO")
    assert ctx["cache_policy"] == CachingQuestion.Kind.FIFO
    warn.assert_called()
    assert any("Invalid cache policy" in str(call.args[0]) for call in warn.call_args_list)
