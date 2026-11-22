# YAML Configuration Guide

This guide documents the minimal YAML schema used by `quizgen`.

## Top-Level Fields

- `name` (string): Quiz title.
- `description` (string, optional): Rendered in Canvas as HTML.
- `practice` (bool, optional): Marks Canvas quizzes as practice.
- `sort order` (list of strings, optional): Topic ordering for descriptive reports; questions retain input order.
- `custom_modules` (list of strings, optional): Python modules to import.
- `questions` (mapping or list): Required. Point values mapped to question blocks, or an ordered list of questions.
- `question_order` (string, optional): Deprecated compatibility setting. Questions always retain input order.

## Practice Generation (CLI)

Tag-filtered practice upload can generate one quiz per matching question type:

```bash
quizgen practice cst334 memory --course-id 12345
```

Useful knobs:

- `--practice-match any|all`: tag matching mode.
- `--practice-variations N`: variation count per question group (default `5`).
- `--practice-question-groups N`: repeat each selected question N times per quiz (default `5`).
- `--practice-tag-source explicit|merged|derived`: choose whether matching uses only explicit tags, merged tags (default), or only derived tags.

Tag audit helpers:

```bash
quizgen tags list
quizgen tags list --only-missing-explicit --include-questions
quizgen tags explain sched
```

## Questions Block

Mapping format: each key is a point value (number). Each value is a mapping of question names to configs.

List format: each item is a question entry with `name` and `points`. This preserves YAML order by default.

### Question Config Keys

- `class` (string): Question class name (registered).
- `kwargs` (mapping, optional): Question-specific parameters.
- `topic` (string, optional): Topic name (e.g., `memory`, `concurrency`).
- `tags` (string or list of strings, optional): Metadata tags used for filtering/reporting (e.g., `cst334`, `memory`, `practice`).
- `questions_per_page` (positive integer, optional): Per-question allocation of the printable page height, including prompt and answer space. `2` allocates half a page; `3` allocates one third. Requires Typst.
- `extra_pages` (nonnegative integer, default `0`): Blank answer pages immediately after this question. The following question starts on a fresh page. Requires Typst.
- `spacing` (number or preset, optional): Legacy vertical space after question; cannot be combined with `questions_per_page`.
  Presets: `NONE`, `SHORT`, `MEDIUM`, `LONG`, `PAGE`, `EXTRA_PAGE`.
- `seed_group` (string, optional): Questions with the same `seed_group` share one RNG seed per generated PDF.
- `num_subquestions` (int, optional): Used for multipart questions.

Question-specific kwargs are passed through directly to the question class. For example, `Segmentation` supports bit caps:

```yaml
questions:
  8:
    Segmentation:
      class: Segmentation
      kwargs:
        max_bits: 12
        max_virtual_bits: 7
```

For `Segmentation`, `max_bits` is a shortcut for `max_physical_bits`. You can also set `max_physical_bits` explicitly.

Recommended core facets:

- `course:<code>` (for example `course:cst334`)
- `topic:<name>` (for example `topic:memory`)
- `skill:<name>` (for example `skill:round_robin`)

Bare forms (for example `cst334`, `memory`) are accepted and normalized automatically.
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

### Fixed page allocations

```yaml
name: Allocated quiz
questions:
  - name: Short response
    points: 4
    class: FromText
    questions_per_page: 3
    kwargs:
      text: Explain your answer.
  - name: Extended response
    points: 8
    class: FromText
    questions_per_page: 2
    extra_pages: 1
    kwargs:
      text: Show your derivation.
```

These settings belong to each question and may also be placed inside its `kwargs`.
Question groups use the settings on their selected member questions.
Allocations use the page height minus its top and bottom margins. Allocations include padding and a horizontal line above each question, with no side or bottom borders or additional gap. A question that does not fit in the remaining space starts on the next page. Input order is preserved, including across point-value groups; `--optimize-space` is accepted as a deprecated no-op.

The title and name line reserve space above the first question. That space is deducted from the allocation area on every page, keeping box heights consistent while allowing a full set of questions on page one. Additional custom instructions are not yet included in this reservation.

Typst measures the prompt, QR code, and PDF aids during a layout preflight before writing the final PDF. Content that exceeds its allocation fails with the question name and required/available heights. Increase that question's allocation or shorten its content. Legacy `spacing` questions retain their existing sizing behavior and can coexist with allocated questions.

Run allocation checks with `python -m unittest discover -s tests -p test_page_allocations.py` (Typst enables the rendering checks).
