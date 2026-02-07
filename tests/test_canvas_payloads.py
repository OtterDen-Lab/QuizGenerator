from unittest.mock import Mock

from QuizGenerator.generate import _build_canvas_payloads
from QuizGenerator.premade_questions.basic import FromText


def test_canvas_payload_dedupes_variations():
    q = FromText(name="Same", points_value=1.0, text="Static")
    q.possible_variations = 1
    course = Mock()
    quiz = Mock()
    payloads = _build_canvas_payloads(
        q,
        course,
        quiz,
        num_variations=5,
        seed_base=0,
        max_attempts=5
    )
    assert len(payloads) == 1
