"""
Performance instrumentation for quiz generation and Canvas upload pipeline.

This module provides timing instrumentation to identify bottlenecks in:
1. Question generation (refresh, get_body, get_explanation)
2. AST rendering for Canvas
3. Canvas API uploads

Usage:
    from QuizGenerator.performance import PerformanceTracker, timer

    # Use context manager
    with timer("operation_name"):
        # your code here

    # Access metrics
    metrics = PerformanceTracker.get_metrics()
    PerformanceTracker.report_summary()
"""

import time
import statistics
import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

log = logging.getLogger(__name__)

@dataclass
class TimingMetric:
    """Container for timing measurements"""
    operation: str
    duration: float
    question_name: Optional[str] = None
    question_type: Optional[str] = None
    variation_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class PerformanceTracker:
    """Global performance tracking system"""

    _metrics: List[TimingMetric] = []
    _active_timers: Dict[str, float] = {}

    @classmethod
    def start_timer(cls, operation: str, **metadata) -> str:
        """Start a named timer and return a timer ID"""
        timer_id = f"{operation}_{time.time()}"
        cls._active_timers[timer_id] = time.perf_counter()
        return timer_id

    @classmethod
    def end_timer(cls, timer_id: str, operation: str, **metadata) -> float:
        """End a timer and record the metric"""
        if timer_id not in cls._active_timers:
            log.warning(f"Timer {timer_id} not found")
            return 0.0

        start_time = cls._active_timers.pop(timer_id)
        duration = time.perf_counter() - start_time

        metric = TimingMetric(
            operation=operation,
            duration=duration,
            **metadata
        )
        cls._metrics.append(metric)

        log.debug(f"{operation}: {duration:.3f}s {metadata}")
        return duration

    @classmethod
    def record_timing(cls, operation: str, duration: float, **metadata):
        """Record a timing metric directly"""
        metric = TimingMetric(
            operation=operation,
            duration=duration,
            **metadata
        )
        cls._metrics.append(metric)

    @classmethod
    def get_metrics(cls) -> List[TimingMetric]:
        """Get all recorded metrics"""
        return cls._metrics.copy()

    @classmethod
    def get_metrics_by_operation(cls, operation: str) -> List[TimingMetric]:
        """Get metrics filtered by operation name"""
        return [m for m in cls._metrics if m.operation == operation]

    @classmethod
    def clear_metrics(cls):
        """Clear all recorded metrics"""
        cls._metrics.clear()
        cls._active_timers.clear()

    @classmethod
    def get_summary_stats(cls, operation: str) -> Dict[str, float]:
        """Get summary statistics for an operation"""
        metrics = cls.get_metrics_by_operation(operation)
        if not metrics:
            return {}

        durations = [m.duration for m in metrics]
        return {
            'count': len(durations),
            'total': sum(durations),
            'mean': statistics.mean(durations),
            'median': statistics.median(durations),
            'min': min(durations),
            'max': max(durations),
            'std': statistics.stdev(durations) if len(durations) > 1 else 0
        }

    @classmethod
    def report_summary(cls, min_duration: float = 0.001) -> str:
        """Generate a summary report of all operations"""
        operations = set(m.operation for m in cls._metrics)

        report_lines = []
        report_lines.append("=== Performance Summary Report ===")
        report_lines.append(f"Total metrics recorded: {len(cls._metrics)}")
        report_lines.append("")

        # Group by operation
        for operation in sorted(operations):
            stats = cls.get_summary_stats(operation)
            if stats['mean'] < min_duration:  # Skip very fast operations
                continue

            report_lines.append(f"{operation}:")
            report_lines.append(f"  Count: {stats['count']}")
            report_lines.append(f"  Total: {stats['total']:.3f}s")
            report_lines.append(f"  Mean:  {stats['mean']:.3f}s ± {stats['std']:.3f}s")
            report_lines.append(f"  Range: {stats['min']:.3f}s - {stats['max']:.3f}s")
            report_lines.append("")

        report = '\n'.join(report_lines)
        print(report)
        return report

    @classmethod
    def report_detailed(cls, operation: Optional[str] = None) -> str:
        """Generate detailed report showing individual measurements"""
        metrics = cls.get_metrics_by_operation(operation) if operation else cls._metrics

        report_lines = []
        if operation:
            report_lines.append(f"=== Detailed Report: {operation} ===")
        else:
            report_lines.append("=== Detailed Report: All Operations ===")

        # Group by question if available
        by_question = defaultdict(list)
        for metric in metrics:
            key = metric.question_name or "unknown"
            by_question[key].append(metric)

        for question_name, question_metrics in by_question.items():
            report_lines.append(f"\nQuestion: {question_name}")
            for metric in sorted(question_metrics, key=lambda m: m.variation_number or 0):
                metadata_str = ""
                if metric.question_type:
                    metadata_str += f" [{metric.question_type}]"
                if metric.variation_number is not None:
                    metadata_str += f" var#{metric.variation_number}"

                report_lines.append(f"  {metric.operation}: {metric.duration:.3f}s{metadata_str}")

        report = '\n'.join(report_lines)
        print(report)
        return report

@contextmanager
def timer(operation: str, **metadata):
    """Context manager for timing operations"""
    timer_id = PerformanceTracker.start_timer(operation, **metadata)
    try:
        yield
    finally:
        PerformanceTracker.end_timer(timer_id, operation, **metadata)

def timed_method(operation_name: Optional[str] = None):
    """Decorator for timing method calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__name__}"

            # Try to extract question info from self if available
            metadata = {}
            if args and hasattr(args[0], 'name'):
                metadata['question_name'] = getattr(args[0], 'name', None)
            if args and hasattr(args[0], '__class__'):
                metadata['question_type'] = args[0].__class__.__name__

            with timer(op_name, **metadata):
                return func(*args, **kwargs)
        return wrapper
    return decorator