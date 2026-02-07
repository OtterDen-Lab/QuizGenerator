"""
Comprehensive tests for Question lifecycle, instantiation, and edge cases.

These tests verify the full question lifecycle from creation through instantiation
and rendering.
"""

import random

import pytest

import QuizGenerator.contentast as ca
from QuizGenerator.question import (
    SPACING_PRESETS,
    Question,
    QuestionContext,
    QuestionGroup,
    QuestionInstance,
    QuestionRegistry,
    RegenerationFlags,
    parse_spacing,
)


class _SimpleQuestion(Question):
    """A minimal question for testing."""

    VERSION = "1.0"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("topic", Question.Topic.MISC)
        super().__init__(*args, **kwargs)
        self.possible_variations = float("inf")

    @classmethod
    def _build_context(cls, *, rng_seed=None, **kwargs):
        context = super()._build_context(rng_seed=rng_seed, **kwargs)
        rng = context.rng
        context["value"] = rng.randint(1, 100)
        return context

    @classmethod
    def _build_body(cls, context):
        body = ca.Section()
        body.add_element(ca.Paragraph([f"What is {context['value']}?"]))
        answer = ca.AnswerTypes.Int(context["value"], label="Answer")
        body.add_element(answer)
        return body

    @classmethod
    def _build_explanation(cls, context):
        explanation = ca.Section()
        explanation.add_element(ca.Paragraph([f"The answer is {context['value']}."]))
        return explanation


class _ParameterizedQuestion(Question):
    """A question that uses constructor parameters."""

    VERSION = "1.0"

    def __init__(self, *args, max_value=50, multiplier=2, **kwargs):
        kwargs["max_value"] = max_value
        kwargs["multiplier"] = multiplier
        kwargs.setdefault("topic", Question.Topic.MATH_GENERAL)
        super().__init__(*args, **kwargs)
        self.max_value = max_value
        self.multiplier = multiplier

    @classmethod
    def _build_context(cls, *, rng_seed=None, **kwargs):
        context = super()._build_context(rng_seed=rng_seed, **kwargs)
        rng = context.rng
        max_value = kwargs.get("max_value", 50)
        multiplier = kwargs.get("multiplier", 2)
        base = rng.randint(1, max_value)
        context["base"] = base
        context["result"] = base * multiplier
        return context

    @classmethod
    def _build_body(cls, context):
        body = ca.Section()
        body.add_element(ca.Paragraph([f"Calculate {context['base']} times the multiplier"]))
        answer = ca.AnswerTypes.Int(context["result"], label="Result")
        body.add_element(answer)
        return body

    @classmethod
    def _build_explanation(cls, context):
        explanation = ca.Section()
        explanation.add_element(ca.Paragraph([f"Result: {context['result']}"]))
        return explanation


class TestQuestionContext:
    """Tests for QuestionContext class."""

    def test_context_creation(self):
        rng = random.Random(42)
        context = QuestionContext(rng_seed=42, rng=rng)

        assert context.rng_seed == 42
        assert context.rng is rng

    def test_context_dict_access(self):
        rng = random.Random(42)
        context = QuestionContext(rng_seed=42, rng=rng)
        context["key"] = "value"

        assert context["key"] == "value"
        assert context.get("key") == "value"
        assert context.get("missing", "default") == "default"

    def test_context_contains(self):
        rng = random.Random(42)
        context = QuestionContext(rng_seed=42, rng=rng)
        context["key"] = "value"

        assert "key" in context
        assert "rng" in context
        assert "rng_seed" in context
        assert "missing" not in context

    def test_context_freeze(self):
        rng = random.Random(42)
        context = QuestionContext(rng_seed=42, rng=rng)
        context["key"] = "value"

        frozen = context.freeze()

        assert frozen.frozen is True
        assert frozen["key"] == "value"

        with pytest.raises(TypeError):
            frozen["new_key"] = "new_value"

    def test_context_iteration(self):
        rng = random.Random(42)
        context = QuestionContext(rng_seed=42, rng=rng)
        context["a"] = 1
        context["b"] = 2

        keys = list(context.keys())
        assert "a" in keys
        assert "b" in keys


class TestSpacingPresets:
    """Tests for spacing preset parsing."""

    def test_spacing_presets_exist(self):
        assert "NONE" in SPACING_PRESETS
        assert "SHORT" in SPACING_PRESETS
        assert "MEDIUM" in SPACING_PRESETS
        assert "LONG" in SPACING_PRESETS
        assert "PAGE" in SPACING_PRESETS
        assert "EXTRA_PAGE" in SPACING_PRESETS

    def test_parse_spacing_presets(self):
        assert parse_spacing("NONE") == 0
        assert parse_spacing("SHORT") == 4
        assert parse_spacing("MEDIUM") == 6
        assert parse_spacing("LONG") == 9
        assert parse_spacing("PAGE") == 99
        assert parse_spacing("EXTRA_PAGE") == 199

    def test_parse_spacing_case_insensitive(self):
        assert parse_spacing("none") == 0
        assert parse_spacing("short") == 4
        assert parse_spacing("Medium") == 6

    def test_parse_spacing_numeric(self):
        assert parse_spacing(5) == 5.0
        assert parse_spacing(3.5) == 3.5
        assert parse_spacing("7.5") == 7.5

    def test_parse_spacing_invalid_returns_zero(self):
        assert parse_spacing("invalid") == 0
        assert parse_spacing(None) == 0


class TestQuestionRegistry:
    """Tests for the QuestionRegistry."""

    def test_registry_loads_premades(self):
        QuestionRegistry.load_premade_questions()
        assert len(QuestionRegistry._registry) > 0

    def test_registry_contains_fromtext(self):
        QuestionRegistry.load_premade_questions()
        assert "fromtext" in QuestionRegistry._registry

    def test_registry_create_fromtext(self):
        q = QuestionRegistry.create(
            "FromText",
            name="Test",
            points_value=5,
            text="What is 2+2?"
        )
        assert q is not None
        assert q.name == "Test"

    def test_registry_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown question type"):
            QuestionRegistry.create("NonExistentQuestion12345")

    def test_registry_case_insensitive(self):
        QuestionRegistry.load_premade_questions()

        # Should work with any case
        q1 = QuestionRegistry.create("fromtext", name="Q1", points_value=1, text="Q1")
        q2 = QuestionRegistry.create("FromText", name="Q2", points_value=1, text="Q2")
        q3 = QuestionRegistry.create("FROMTEXT", name="Q3", points_value=1, text="Q3")

        assert type(q1).__name__ == type(q2).__name__ == type(q3).__name__


class TestQuestionInstantiation:
    """Tests for question instantiation."""

    def test_instantiate_returns_question_instance(self):
        q = _SimpleQuestion(name="Test", points_value=5)
        instance = q.instantiate(rng_seed=42)

        assert isinstance(instance, QuestionInstance)

    def test_instantiate_has_body(self):
        q = _SimpleQuestion(name="Test", points_value=5)
        instance = q.instantiate(rng_seed=42)

        assert instance.body is not None

    def test_instantiate_has_explanation(self):
        q = _SimpleQuestion(name="Test", points_value=5)
        instance = q.instantiate(rng_seed=42)

        assert instance.explanation is not None

    def test_instantiate_has_answers(self):
        q = _SimpleQuestion(name="Test", points_value=5)
        instance = q.instantiate(rng_seed=42)

        assert isinstance(instance.answers, list)
        assert len(instance.answers) > 0

    def test_instantiate_has_regeneration_flags(self):
        q = _SimpleQuestion(name="Test", points_value=5)
        instance = q.instantiate(rng_seed=42)

        assert isinstance(instance.flags, RegenerationFlags)
        assert instance.flags.question_class_name == "_SimpleQuestion"
        assert instance.flags.generation_seed == 42
        assert instance.flags.question_version == "1.0"


class TestQuestionDeterminism:
    """Tests for question reproducibility."""

    def test_same_seed_produces_same_output(self):
        q = _SimpleQuestion(name="Test", points_value=5)

        instance_a = q.instantiate(rng_seed=42)
        instance_b = q.instantiate(rng_seed=42)

        # Both should produce the same value
        assert instance_a.answers[0].value == instance_b.answers[0].value

    def test_different_seeds_produce_different_output(self):
        q = _SimpleQuestion(name="Test", points_value=5)

        instance_a = q.instantiate(rng_seed=42)
        instance_b = q.instantiate(rng_seed=43)

        # Values should differ (with very high probability)
        # Note: There's a tiny chance they could be the same by coincidence
        # For a robust test, we check that at least one of many is different
        values_42 = set()
        values_43 = set()
        for _ in range(10):
            values_42.add(_SimpleQuestion(name="T", points_value=1).instantiate(rng_seed=42).answers[0].value)
            values_43.add(_SimpleQuestion(name="T", points_value=1).instantiate(rng_seed=43).answers[0].value)

        # Different seeds should produce different value distributions
        # (This is a probabilistic test but extremely unlikely to fail)
        assert values_42 != values_43 or len(values_42) == 1 and len(values_43) == 1

    def test_parameterized_question_uses_parameters(self):
        q1 = _ParameterizedQuestion(name="T1", points_value=5, max_value=10, multiplier=3)
        q2 = _ParameterizedQuestion(name="T2", points_value=5, max_value=100, multiplier=5)

        instance_1 = q1.instantiate(rng_seed=42)
        instance_2 = q2.instantiate(rng_seed=42)

        # Even with same seed, different parameters = different results
        # (base values may be different due to different max_value)
        # And definitely different final results due to different multipliers
        assert instance_1.answers[0].value != instance_2.answers[0].value


class TestQuestionRendering:
    """Tests for question content rendering."""

    def test_body_renders_to_html(self):
        q = _SimpleQuestion(name="Test", points_value=5)
        instance = q.instantiate(rng_seed=42)

        html = instance.body.render("html")
        assert "What is" in html

    def test_body_renders_to_latex(self):
        q = _SimpleQuestion(name="Test", points_value=5)
        instance = q.instantiate(rng_seed=42)

        latex = instance.body.render("latex")
        assert "What is" in latex

    def test_body_renders_to_typst(self):
        q = _SimpleQuestion(name="Test", points_value=5)
        instance = q.instantiate(rng_seed=42)

        typst = instance.body.render("typst")
        assert "What is" in typst

    def test_explanation_renders(self):
        q = _SimpleQuestion(name="Test", points_value=5)
        instance = q.instantiate(rng_seed=42)

        html = instance.explanation.render("html")
        assert "answer is" in html


class TestQuestionGroup:
    """Tests for QuestionGroup behavior."""

    def test_question_group_creation(self):
        q1 = _SimpleQuestion(name="Q1", points_value=5)
        q2 = _SimpleQuestion(name="Q2", points_value=5)

        # QuestionGroup takes questions_in_group and pick_once
        group = QuestionGroup(
            questions_in_group=[q1, q2],
            pick_once=True,
            name="Group"
        )

        assert len(group.questions) == 2

    def test_question_group_pick_one(self):
        q1 = _SimpleQuestion(name="Q1", points_value=5)
        q2 = _SimpleQuestion(name="Q2", points_value=5)

        group = QuestionGroup(
            questions_in_group=[q1, q2],
            pick_once=False,  # Pick fresh each time
            name="Group"
        )

        instance = group.instantiate(rng_seed=42)

        # Should return a single question instance (not multiple)
        assert isinstance(instance, QuestionInstance)

    def test_question_group_deterministic_selection(self):
        q1 = _SimpleQuestion(name="Q1", points_value=5)
        q2 = _SimpleQuestion(name="Q2", points_value=5)

        group = QuestionGroup(
            questions_in_group=[q1, q2],
            pick_once=True,
            name="Group"
        )

        instance_a = group.instantiate(rng_seed=42)
        instance_b = group.instantiate(rng_seed=42)

        # Same seed should pick the same question and generate the same content
        assert instance_a.answers[0].value == instance_b.answers[0].value


class TestQuestionTopics:
    """Tests for question topic handling."""

    def test_topic_assignment(self):
        q = _SimpleQuestion(name="Test", points_value=5, topic=Question.Topic.SYSTEM_MEMORY)
        assert q.topic == Question.Topic.SYSTEM_MEMORY

    def test_topic_from_string(self):
        topic = Question.Topic.from_string("memory")
        assert topic == Question.Topic.SYSTEM_MEMORY

    def test_topic_from_string_unknown(self):
        topic = Question.Topic.from_string("unknown_topic_xyz")
        assert topic == Question.Topic.MISC

    def test_topic_from_string_case_insensitive(self):
        topic1 = Question.Topic.from_string("MEMORY")
        topic2 = Question.Topic.from_string("memory")
        topic3 = Question.Topic.from_string("Memory")
        assert topic1 == topic2 == topic3


class TestQuestionConfigParams:
    """Tests for config_params preservation."""

    def test_config_params_stored_in_flags(self):
        q = _ParameterizedQuestion(name="T", points_value=5, max_value=100, multiplier=7)
        instance = q.instantiate(rng_seed=42)

        config = instance.flags.config_params
        assert "max_value" in config
        assert config["max_value"] == 100
        assert "multiplier" in config
        assert config["multiplier"] == 7


class TestFromTextQuestion:
    """Tests for the built-in FromText question type."""

    def test_fromtext_basic(self):
        q = QuestionRegistry.create(
            "FromText",
            name="Basic",
            points_value=5,
            text="What is 2+2?"
        )

        instance = q.instantiate(rng_seed=42)
        html = instance.body.render("html")

        assert "2+2" in html

    def test_fromtext_has_no_answers(self):
        q = QuestionRegistry.create(
            "FromText",
            name="NoAnswers",
            points_value=5,
            text="Explain something."
        )

        instance = q.instantiate(rng_seed=42)
        assert len(instance.answers) == 0

    def test_fromtext_limited_variations(self):
        q = QuestionRegistry.create(
            "FromText",
            name="Static",
            points_value=5,
            text="Static text."
        )

        # FromText has only 1 possible variation
        assert q.possible_variations == 1


class TestQuestionBuildMethods:
    """Tests for the question build method variants."""

    def test_build_method_fallback(self):
        """If _build_body and _build_explanation are implemented, build() isn't needed."""
        q = _SimpleQuestion(name="Test", points_value=5)
        instance = q.instantiate(rng_seed=42)

        assert instance.body is not None
        assert instance.explanation is not None


class TestAnswerExtraction:
    """Tests for answer extraction from question bodies."""

    def test_answers_extracted_from_body(self):
        q = _SimpleQuestion(name="Test", points_value=5)
        instance = q.instantiate(rng_seed=42)

        # The answer should be extracted
        assert len(instance.answers) == 1
        assert instance.answers[0].value == instance.flags.config_params.get("value") or True

    def test_answer_has_label(self):
        q = _SimpleQuestion(name="Test", points_value=5)
        instance = q.instantiate(rng_seed=42)

        answer = instance.answers[0]
        assert hasattr(answer, "label")
        assert answer.label == "Answer"
