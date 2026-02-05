import QuizGenerator.contentast as ca
from QuizGenerator.question import Question


class _DummyQuestion(Question):
    VERSION = "1.0"

    @classmethod
    def _build_context(cls, *, rng_seed=None, **kwargs):
        context = super()._build_context(rng_seed=rng_seed, **kwargs)
        context["value"] = context.rng.randint(1, 100)
        return context

    @classmethod
    def _build_body(cls, context):
        body = ca.Section()
        body.add_element(ca.Paragraph([f"Value: {context['value']}"]))
        answer = ca.AnswerTypes.Int(context["value"], label="Value")
        body.add_element(answer)
        return body

    @classmethod
    def _build_explanation(cls, context):
        explanation = ca.Section()
        explanation.add_element(ca.Paragraph([f"Answer: {context['value']}"]))
        return explanation


def test_question_flow_determinism():
    q1 = _DummyQuestion(name="Dummy", points_value=1.0)
    q2 = _DummyQuestion(name="Dummy", points_value=1.0)

    instance_a = q1.instantiate(rng_seed=123)
    instance_b = q2.instantiate(rng_seed=123)
    assert instance_a.answers[0].value == instance_b.answers[0].value

    instance_c = q1.instantiate(rng_seed=124)
    assert instance_c.answers[0].value != instance_a.answers[0].value
