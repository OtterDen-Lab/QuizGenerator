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
quizgen --yaml example_files/example_exam.yaml --num_pdfs 1
```

PDFs are generated in `out/`.

## Canvas Upload (Optional)

Create `~/.env` or pass `--env`:

```
CANVAS_API_URL=https://canvas.instructure.com
CANVAS_API_KEY=your_api_key_here
```

Then:

```bash
quizgen --yaml example_files/example_exam.yaml --num_canvas 5 --course_id 12345
```

## Testing Mode (No YAML Needed)

```bash
quizgen --test_all 5
```

Skip questions that require optional extras:

```bash
quizgen --test_all 5 --skip_missing_extras
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

## Troubleshooting

- **Pandoc missing**: ContentAST will fall back to raw markdown for LaTeX/Typst.
  Install pandoc to improve rendering quality.
