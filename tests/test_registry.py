from QuizGenerator.question import QuestionRegistry


def test_registry_load_premades_does_not_crash():
    QuestionRegistry.load_premade_questions()
    assert QuestionRegistry._registry
