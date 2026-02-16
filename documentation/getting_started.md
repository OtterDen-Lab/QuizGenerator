# Getting Started

This guide covers a minimal setup to generate PDFs and (optionally) upload to Canvas.

## Install

```bash
pip install QuizGenerator
```

For local development:

```bash
pip install -e .
```

## Reproducible installs (recommended)

For a fully pinned environment, use the lockfile:

```bash
uv sync --locked
```

## Requirements

- Python 3.12+
- Typst (default PDF renderer)
- LaTeX (optional; use `--latex`)
- Pandoc (recommended for markdown conversion in ContentAST)

## Optional Extras

```bash
# QR code grading (image scanning)
pip install QuizGenerator[grading]

# CST463 machine learning question set
pip install QuizGenerator[cst463]
```

## Quick Start (PDF)

```bash
quizgen generate --yaml example_files/example_exam.yaml --num_pdfs 1
```

PDFs are generated in `out/`.

## Dependency Check (Optional)

Verify external tools before generating:

```bash
quizgen deps
```

## Ordered YAML (Optional)

If you want to preserve question order, use the list format:

```yaml
name: "Midterm Exam"
question_order: yaml
questions:
  - name: "Process Scheduling"
    points: 10
    class: FIFOScheduling
  - name: "Memory Paging"
    points: 5
    class: PagingQuestion
```

## Layout Optimization (Optional)

To reduce PDF page count by reordering questions, use:

```bash
quizgen generate --yaml example_files/example_exam.yaml --num_pdfs 1 --optimize_space
```

This affects both PDF and Canvas order.

## Canvas Upload (Optional)

Create `~/.env` or pass `--env`:

```
CANVAS_API_URL=https://canvas.instructure.com
CANVAS_API_KEY=your_api_key_here
```

Then:

```bash
quizgen generate --yaml example_files/example_exam.yaml --num_canvas 5 --course_id 12345
```

## Testing Mode (No YAML Needed)

```bash
quizgen test 5
```

Skip questions that require optional extras:

```bash
quizgen test 5 --skip_missing_extras
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## Troubleshooting

- **Pandoc missing**: ContentAST will fall back to raw markdown for LaTeX/Typst.
  Install pandoc to improve rendering quality.
