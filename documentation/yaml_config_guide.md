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

## Practice Generation (CLI)

Tag-filtered practice upload can generate one quiz per matching question type:

```bash
quizgen --generate_practice cst334 memory --course_id 12345
```

Useful knobs:

- `--practice_match any|all`: tag matching mode.
- `--practice_variations N`: variation count per question group (default `5`).
- `--practice_question_groups N`: repeat each selected question N times per quiz (default `5`).

## Questions Block

Mapping format: each key is a point value (number). Each value is a mapping of question names to configs.

List format: each item is a question entry with `name` and `points`. This preserves YAML order by default.

### Question Config Keys

- `class` (string): Question class name (registered).
- `kwargs` (mapping, optional): Question-specific parameters.
- `topic` (string, optional): Topic name (e.g., `memory`, `concurrency`).
- `tags` (string or list of strings, optional): Metadata tags used for filtering/reporting (e.g., `cst334`, `memory`, `practice`).
- `spacing` (number or preset, optional): Vertical space after question.
  Presets: `NONE`, `SHORT`, `MEDIUM`, `LONG`, `PAGE`, `EXTRA_PAGE`.
- `seed_group` (string, optional): Questions with the same `seed_group` share one RNG seed per generated PDF.
- `num_subquestions` (int, optional): Used for multipart questions.

Recommended core facets:

- `course:<code>` (for example `course:cst334`)
- `topic:<name>` (for example `topic:memory`)
- `skill:<name>` (for example `skill:round_robin`)

Bare legacy forms (for example `cst334`, `memory`) are still accepted and are normalized automatically.
Free-form tags are allowed.

### Per-Question `_config`

Used to enable grouping and repeats:

- `group` (bool): Treat nested entries as a question group.
- `num_to_pick` (int): Number to pick from a group (currently 1).
- `random_per_student` (bool): If true, group selection varies per student.
- `repeat` (int): Repeat a question N times (with seed offsets).
- `topic` (string): Topic for all grouped questions.
- `tags` (string or list of strings): Tags applied to grouped questions.

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
      tags: [course:cst334, topic:processes, practice]
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
