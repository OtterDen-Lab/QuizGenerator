from unittest.mock import Mock

from QuizGenerator.generate import _build_canvas_payloads, _normalize_canvas_html
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


def test_normalize_canvas_html_preserves_non_hashed_urls():
    html = (
        '<p>See <a href="https://example.com/doc">doc</a></p>'
        '<img src="/files/1/preview" alt="nohash">'
    )
    normalized = _normalize_canvas_html(html)
    assert 'href="https://example.com/doc"' in normalized
    assert 'src="/files/1/preview"' in normalized


def test_normalize_canvas_html_strips_hashed_img_src():
    html = (
        '<img src="/files/123/preview" data-quizgen-hash="abc123" alt="hashed">'
    )
    normalized = _normalize_canvas_html(html)
    assert 'data-quizgen-hash="abc123"' in normalized
    assert 'src="__URL__"' in normalized
