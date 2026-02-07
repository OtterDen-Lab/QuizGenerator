# Contributing

Thanks for helping improve QuizGenerator! This repo is used by instructors, so changes should prioritize correctness and reproducibility.

## Quick Start

```bash
pip install -e ".[dev]"
```

## Local Checks

- Lint: `ruff check --fix`
- Smoke PDF: `quizgen --yaml example_files/example_exam.yaml --num_pdfs 1`
- Test all questions: `quizgen --test_all 2 --skip_missing_extras`

## Question Authoring

- Implement `_build_context`, `_build_body`, `_build_explanation` on every question class.
- Use `context.rng` for deterministic randomness.
- Avoid `refresh()`; it has been removed.
- Prefer ContentAST elements (no raw LaTeX/HTML).

## Docs

- Update `documentation/` if you add or change CLI flags or YAML options.
- Add an example config if behavior changes are user‑visible.

