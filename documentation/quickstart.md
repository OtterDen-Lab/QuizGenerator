# Quick Start

## Install

```bash
pip install QuizGenerator
```

Optional extras:

```bash
pip install QuizGenerator[grading]
pip install QuizGenerator[cst463]
```

## Generate a PDF

```bash
quizgen --yaml example_files/example_exam.yaml --num_pdfs 1
```

PDFs are written to `out/`.

## Push to Canvas

Create `~/.env` (or pass `--env`) with:

```
CANVAS_API_URL=https://canvas.instructure.com
CANVAS_API_KEY=your_api_key_here
```

Then:

```bash
quizgen --yaml example_files/example_exam.yaml --num_canvas 5 --course_id 12345
```

## Test All Questions

```bash
quizgen --test_all 2
```

Skip questions that require optional extras:

```bash
quizgen --test_all 2 --skip_missing_extras
```
