#!/usr/bin/env python
"""
Test script to exercise the performance instrumentation system without
actually uploading to Canvas.
"""

from QuizGenerator.quiz import Quiz
from QuizGenerator.performance import PerformanceTracker

def test_performance_instrumentation():
    print("Testing performance instrumentation...")

    # Clear any previous metrics
    PerformanceTracker.clear_metrics()

    # Load a quiz
    quizzes = Quiz.from_yaml("example_files/scratch.yaml")
    quiz = quizzes[0]

    # Generate some questions to exercise the timing code
    print("Generating questions...")
    for i in range(3):
        print(f"  Generating question {i+1}/3")
        quiz_doc = quiz.get_quiz(rng_seed=i)

        # Render to HTML (to exercise AST rendering)
        for element in quiz_doc.elements:
            if hasattr(element, 'body'):
                element.body.render("html")
            if hasattr(element, 'explanation'):
                element.explanation.render("html")

    # Generate performance report
    print("\n" + "="*80)
    print("PERFORMANCE TEST REPORT")
    print("="*80)

    metrics = PerformanceTracker.get_metrics()
    print(f"Total metrics recorded: {len(metrics)}")

    if metrics:
        PerformanceTracker.report_summary(min_duration=0.001)

        print("\nDetailed breakdown:")
        for metric in metrics[:10]:  # Show first 10 metrics
            print(f"  {metric.operation}: {metric.duration:.4f}s ({metric.question_name}, {metric.question_type})")
    else:
        print("No metrics recorded. This suggests the instrumentation may not be working.")

if __name__ == "__main__":
    test_performance_instrumentation()