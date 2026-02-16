import math
import random

from QuizGenerator.premade_questions.cst334.process import MLFQQuestion
from QuizGenerator.question import QuestionContext


def _build_context(*, rng_seed: int) -> QuestionContext:
    ctx = QuestionContext(rng_seed=rng_seed, rng=random.Random(rng_seed))
    ctx["num_queues"] = 3
    ctx["job_stats"] = {
        0: {"arrival_time": 0, "TAT": 11.2},
        1: {"arrival_time": 3, "TAT": 6.1},
    }
    return ctx


def test_pdf_aid_time_horizon_is_masked_and_block_aligned():
    ctx = _build_context(rng_seed=123)
    horizon = MLFQQuestion._get_pdf_aid_time_horizon(ctx)

    true_horizon = int(
        math.ceil(
            max(
                row["arrival_time"] + row["TAT"]
                for row in ctx["job_stats"].values()
            )
        )
    )
    rounded = int(math.ceil(true_horizon / MLFQQuestion.AID_TIME_BLOCK) * MLFQQuestion.AID_TIME_BLOCK)

    assert horizon % MLFQQuestion.AID_TIME_BLOCK == 0
    assert horizon >= rounded + (MLFQQuestion.AID_BUFFER_BLOCKS[0] * MLFQQuestion.AID_TIME_BLOCK)
    assert horizon <= rounded + (MLFQQuestion.AID_BUFFER_BLOCKS[1] * MLFQQuestion.AID_TIME_BLOCK)


def test_pdf_aid_time_horizon_is_deterministic_for_seed():
    h1 = MLFQQuestion._get_pdf_aid_time_horizon(_build_context(rng_seed=456))
    h2 = MLFQQuestion._get_pdf_aid_time_horizon(_build_context(rng_seed=456))
    assert h1 == h2
