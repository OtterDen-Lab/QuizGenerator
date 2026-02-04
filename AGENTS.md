# Repository Guidelines

## Project Structure & Module Organization

- `QuizGenerator/` is the main Python package (question types, quiz generation, Canvas integration).
- `quizgen` provides the CLI entry point; the package also exposes `quizregen`.
- `documentation/` contains user guides and configuration references.
- `example_files/` and `examples/` hold sample YAML configs and outputs.
- `scripts/` includes maintenance utilities (see `scripts/README.md`).
- `out/` is the default output directory for generated PDFs and artifacts.

## Build, Test, and Development Commands

- `pip install -e .` — install the package in editable mode for local development.
- `quizgen --help` — show CLI options.
- `quizgen --yaml example_files/sample.yaml --num_pdfs 3` — generate PDFs in `out/` (Typst is the default renderer).
- `quizgen --latex --num_pdfs 3` — force LaTeX rendering when needed.
- `python scripts/vendor_lms_interface.py --dry-run` — preview LMSInterface vendoring changes.

## Coding Style & Naming Conventions

- Python 3.12+; follow PEP 8 with 4-space indentation.
- Use `snake_case` for functions/variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.
- Keep modules focused; new question types belong under `QuizGenerator/premade_questions/` or a custom module referenced in YAML.

## Question Authoring Pattern

- Implement `_build_context`, `_build_body`, and `_build_explanation` on every question class.
- `_build_context` should seed and return a `QuestionContext`; use `context.rng` (or `context["rng"]`) for deterministic randomness.
- Add extra data via `context["key"] = value` (stored in `context.data`); keep instance state immutable.
- `_build_body` and `_build_explanation` take `context` and should avoid mutating instance state.
- Do not implement or call `refresh()`; it has been removed.

## Testing Guidelines

- No dedicated test suite is currently committed.
- For changes, run a manual smoke check by generating a small quiz:
  - `quizgen --yaml example_files/sample.yaml --num_pdfs 1`
- If adding tests, prefer `tests/` with filenames like `test_<area>.py` and document how to run them.

## Commit & Pull Request Guidelines

- Commit messages in this repo are short, sentence-case summaries (e.g., “Fixing rounding in round robin”).
- Keep commits focused and explain *why* when the change is non-obvious.
- PRs should include:
  - A brief description of the change and rationale.
  - Example output or screenshots for rendering/formatting changes.
  - Any new CLI flags or YAML options documented in `documentation/`.

## Security & Configuration Tips

- Canvas credentials are read from `~/.env` (`CANVAS_API_URL`, `CANVAS_API_KEY`).
- PDF generation uses Typst by default; LaTeX is legacy and may be removed.
- Avoid committing generated files in `out/` or local secrets.

## Agent Notes Policy

- Store any agent-generated Markdown notes in `agent_notes/` at the repo root.
