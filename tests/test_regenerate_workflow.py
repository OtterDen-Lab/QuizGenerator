import QuizGenerator.contentast as ca
from QuizGenerator.question import Question, QuestionRegistry
from QuizGenerator.regenerate import (
    regenerate_from_metadata,
    regenerate_from_yaml_metadata,
)


@QuestionRegistry.register("test.regen.seeded")
class _SeededRegenerationQuestion(Question):
    VERSION = "1.0"

    @classmethod
    def _build_context(cls, *, rng_seed=None, **kwargs):
        context = super()._build_context(rng_seed=rng_seed, **kwargs)
        context["value"] = int(context.rng_seed or 0) + 17
        return context

    @classmethod
    def _build_body(cls, context):
        body = ca.Section()
        body.add_element(ca.Paragraph([f"Value: {context['value']}"]))
        body.add_element(ca.AnswerTypes.Int(context["value"], label="Value"))
        return body

    @classmethod
    def _build_explanation(cls, context):
        explanation = ca.Section()
        explanation.add_element(ca.Paragraph([f"Answer: {context['value']}"]))
        return explanation


def test_regenerate_from_metadata_uses_seeded_output():
    result_a = regenerate_from_metadata(
        question_type="test.regen.seeded",
        seed=42,
        version="1.0",
        points=1.0,
    )
    result_b = regenerate_from_metadata(
        question_type="test.regen.seeded",
        seed=42,
        version="1.0",
        points=1.0,
    )
    result_c = regenerate_from_metadata(
        question_type="test.regen.seeded",
        seed=43,
        version="1.0",
        points=1.0,
    )

    assert result_a["answer_objects"][0].value == result_b["answer_objects"][0].value
    assert result_c["answer_objects"][0].value != result_a["answer_objects"][0].value


def test_regenerate_from_yaml_metadata_reports_mismatches():
    yaml_text = """
name: "Seeded YAML Quiz"
yaml_id: "yaml-alpha"
questions:
  2:
    "Seeded question":
      question_id: "seeded-q1"
      class: test.regen.seeded
"""

    result = regenerate_from_yaml_metadata(
        question_id="seeded-q1",
        seed=42,
        points=1.0,
        yaml_id="yaml-beta",
        yaml_text=yaml_text,
    )

    warnings = result.get("warnings", [])
    assert any("YAML id mismatch" in warning for warning in warnings)
    assert any("Points mismatch" in warning for warning in warnings)
    assert result["answer_objects"][0].value == 59
