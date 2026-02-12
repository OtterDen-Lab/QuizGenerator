# YAML Configuration Guide

This guide documents the minimal YAML schema used by `quizgen`.

## Top-Level Fields

- `name` (string): Quiz title.
- `description` (string, optional): Rendered in Canvas as HTML.
- `practice` (bool, optional): Marks Canvas quizzes as practice.
- `sort order` (list of strings, optional): Topic ordering for PDF/Canvas.
- `custom_modules` (list of strings, optional): Python modules to import.
- `questions` (mapping or list): Required. Point values mapped to question blocks, or an ordered list of questions.
- `question_order` (string, optional): Ordering mode. `yaml` preserves YAML order, `points` groups by point value.

## Questions Block

Mapping format: each key is a point value (number). Each value is a mapping of question names to configs.

List format: each item is a question entry with `name` and `points`. This preserves YAML order by default.

### Question Config Keys

- `class` (string): Question class name (registered).
- `kwargs` (mapping, optional): Question-specific parameters.
- `topic` (string, optional): Topic name (e.g., `memory`, `concurrency`).
- `spacing` (number or preset, optional): Vertical space after question.
  Presets: `NONE`, `SHORT`, `MEDIUM`, `LONG`, `PAGE`, `EXTRA_PAGE`.
- `seed_group` (string, optional): Questions with the same `seed_group` share one RNG seed per generated PDF.
- `num_subquestions` (int, optional): Used for multipart questions.

### Per-Question `_config`

Used to enable grouping and repeats:

- `group` (bool): Treat nested entries as a question group.
- `num_to_pick` (int): Number to pick from a group (currently 1).
- `random_per_student` (bool): If true, group selection varies per student.
- `repeat` (int): Repeat a question N times (with seed offsets).
- `topic` (string): Topic for all grouped questions.

**Note:** Legacy keys `pick` and `repeat` at the question level are no longer supported. Use `_config` instead.

### Per-Point `_config`

Used for layout rules:

- `preserve_order` (bool): If true, preserves question order within that point tier.

## Example

```yaml
name: "CST 334 Exam"
description: "Midterm coverage: memory, scheduling."
practice: false
sort order: [memory, processes, concurrency]
custom_modules:
  - my_questions

questions:
  10:
    _config:
      preserve_order: true
    "Scheduling":
      class: SchedulingQuestion
      kwargs:
        spacing: MEDIUM

  5:
    "Text Question":
      class: FromText
      kwargs:
        text: "Explain paging and segmentation."
    "Grouped Question":
      _config:
        group: true
        random_per_student: false
        topic: memory
      "Paging":
        class: PagingQuestion
      "TLB":
        class: TLBQuestion
```

## List Format Example (Preserves Order)

```yaml
name: "CST 334 Exam"
question_order: yaml
questions:
  - name: "Scheduling"
    points: 10
    class: SchedulingQuestion

  - name: "Text Question"
    points: 5
    class: FromText
    kwargs:
      text: "Explain paging and segmentation."
```
