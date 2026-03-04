import collections

from QuizGenerator.generation.premade_questions.cst334.memory_questions import (
    CachingQuestion,
)


def test_popular_workload_biases_toward_low_pages():
    context = CachingQuestion._build_context(
        rng_seed=17,
        policy="FIFO",
        workload="popular",
        num_elements=8,
        cache_size=3,
        num_requests=60,
    )
    counts = collections.Counter(context["requests"])
    assert counts[0] > counts[7]


def test_capacity_miss_threshold_for_interesting():
    assert CachingQuestion.is_interesting_ctx(
        {"num_capacity_misses": 1, "can_have_capacity_miss": True}
    )
    assert not CachingQuestion.is_interesting_ctx(
        {"num_capacity_misses": 0, "can_have_capacity_miss": True}
    )
    assert CachingQuestion.is_interesting_ctx(
        {"num_capacity_misses": 0, "can_have_capacity_miss": False}
    )
