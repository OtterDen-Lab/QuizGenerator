# QuizGenerator

Generate randomized quiz questions for Canvas LMS and PDF exams with support for multiple question types, automatic variation generation, and QR code-based answer keys.

## Features

- **Multiple Output Formats**: Generate PDFs (LaTeX or Typst) and Canvas LMS quizzes
- **Automatic Variations**: Create unique versions for each student
- **Extensible**: Plugin system for custom question types
- **Built-in Question Library**: Memory management, process scheduling, calculus, linear algebra, and more
- **QR Code Answer Keys**: Regenerate exact exam versions from QR codes
- **Canvas Integration**: Direct upload to Canvas with variation support

## Installation

```bash
pip install QuizGenerator
```

### Reproducible installs (recommended)

If you want a fully pinned environment for a semester, use the lockfile:

```bash
uv sync --locked
```

We keep dependency ranges in `pyproject.toml` for flexibility and rely on `uv.lock`
to pin exact versions when you need reproducible builds.

### System Requirements

- Python 3.12+
- [Typst](https://typst.app/) (default PDF renderer)
- Optional: LaTeX distribution with `latexmk` (if using `--latex`)
- Recommended: [Pandoc](https://pandoc.org/) (for markdown conversion)
- Optional (LaTeX + QR codes): [Inkscape](https://inkscape.org/) for SVG conversion

### Optional Dependencies

```bash
# For QR code grading support
pip install "QuizGenerator[grading]"

# For CST463 machine learning questions
pip install "QuizGenerator[cst463]"
```

## Quick Start

Need a 2‑minute setup? See `documentation/getting_started.md`.

### 1. Create a quiz configuration (YAML)

```yaml
# my_quiz.yaml
name: "Midterm Exam"

questions:
  10:  # 10-point questions
    "Process Scheduling":
      class: FIFOScheduling

  5:   # 5-point questions
    "Memory Paging":
      class: PagingQuestion

    "Vector Math":
      class: VectorAddition
```

You can also provide an ordered list of questions:

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

### 2. Generate PDFs

```bash
quizgen --yaml my_quiz.yaml --num_pdfs 3
```

PDFs will be created in the `out/` directory.

### 3. Upload to Canvas

```bash
# Set up Canvas credentials in ~/.env first:
# CANVAS_API_URL=https://canvas.instructure.com
# CANVAS_API_KEY=your_api_key_here

quizgen \
  --yaml my_quiz.yaml \
  --num_canvas 5 \
  --course_id 12345
```

## Creating Custom Questions

QuizGenerator supports two approaches for adding custom question types:

### Option 1: Entry Points (Recommended for Distribution)

Create a pip-installable package:

```toml
# pyproject.toml
[project.entry-points."quizgenerator.questions"]
my_question = "my_package.questions:MyCustomQuestion"
```

After `pip install`, your questions are automatically available!

### Option 2: Direct Import (Quick & Easy)

Add to your quiz YAML:

```yaml
custom_modules:
  - my_questions  # Import my_questions.py

questions:
  10:
    "My Question":
      class: MyCustomQuestion
```

See [documentation/custom_questions.md](documentation/custom_questions.md) for complete guide.

### Question Authoring Pattern (New)

All questions follow the same three‑method flow:

```python
class MyQuestion(Question):
    @classmethod
    def _build_context(cls, *, rng_seed=None, **kwargs):
        context = super()._build_context(rng_seed=rng_seed, **kwargs)
        rng = context.rng
        context["value"] = rng.randint(1, 10)
        return context

    @classmethod
    def _build_body(cls, context):
        body = ca.Section()
        body.add_element(ca.Paragraph([f"Value: {context['value']}"]))
        body.add_element(ca.AnswerTypes.Int(context["value"], label="Value"))
        return body

    @classmethod
    def _build_explanation(cls, context):
        explanation = ca.Section()
        explanation.add_element(ca.Paragraph([f"Answer: {context['value']}"]))
        return explanation
```

Notes:
- Always use `context.rng` (or `context["rng"]`) for deterministic randomness.
- Avoid `refresh()`; it is no longer part of the API.

## Built-in Question Types

### Operating Systems (CST334)
- `FIFOScheduling`, `SJFScheduling`, `RoundRobinScheduling`
- `PagingQuestion`, `TLBQuestion`
- `SemaphoreQuestion`, `MutexQuestion`

### Machine Learning / Math (CST463)
- `VectorAddition`, `VectorDotProduct`, `VectorMagnitude`
- `MatrixAddition`, `MatrixMultiplication`, `MatrixTranspose`
- `DerivativeBasic`, `DerivativeChain`
- `GradientDescentStep`

### General
- `FromText` - Custom text questions
- `FromGenerator` - Programmatically generated questions (requires `--allow_generator` or `QUIZGEN_ALLOW_GENERATOR=1`)

## Documentation

- [Getting Started Guide](documentation/getting_started.md)
- [First 5 Minutes](documentation/first_5_minutes.md)
- [Custom Questions Guide](documentation/custom_questions.md)
- [YAML Configuration Reference](documentation/yaml_config_guide.md)

## Canvas Setup

1. Create a `~/.env` file with your Canvas credentials:

```bash
# For testing/development
CANVAS_API_URL=https://canvas.test.instructure.com
CANVAS_API_KEY=your_test_api_key

# For production
CANVAS_API_URL_prod=https://canvas.instructure.com
CANVAS_API_KEY_prod=your_prod_api_key
```

2. Use `--prod` flag for production Canvas instance:

```bash
quizgen --prod --num_canvas 5 --course_id 12345
```

## Advanced Features

### Typst Support

Typst is the default for faster compilation. Use `--latex` to force LaTeX:

```bash
quizgen --latex --num_pdfs 3
```

Experimental: `--typst_measurement` uses Typst to measure question height for tighter layout.
It can change pagination and ordering, so use with care on finalized exams.

### Layout Optimization

By default, questions keep their YAML order (or point-value ordering for mapping format).
Use `--optimize_space` to reorder questions to reduce PDF page count. This also affects Canvas order.

### Deterministic Generation

Use seeds for reproducible quizzes:

```bash
quizgen --seed 42 --num_pdfs 3
```

### Generation Controls

Limit backoff attempts for questions that retry until they are "interesting":

```bash
quizgen --yaml my_quiz.yaml --num_pdfs 1 --max_backoff_attempts 50
```

Set a default numeric tolerance for float answers (overridable per question):

```bash
quizgen --yaml my_quiz.yaml --num_pdfs 1 --float_tolerance 0.01
```

Per-answer override in custom questions:

```python
ca.AnswerTypes.Float(value, label="Result", tolerance=0.005)
```

### QR Code Regeneration

Each generated exam includes a QR code that stores:
- Question types and parameters
- Random seed
- Version information

Use the grading tools to scan QR codes and regenerate exact exam versions.

## Security Considerations

### FromGenerator Warning

The `FromGenerator` question type executes **arbitrary Python code** from your YAML configuration files. This is a powerful feature for creating dynamic questions, but it carries security risks:

- **Only use `FromGenerator` with YAML files you completely trust**
- Never run `--allow_generator` on YAML files from untrusted sources
- Be cautious when sharing question banks that contain generator code

`FromGenerator` is disabled by default. To enable it, use one of:
```bash
quizgen --allow_generator --yaml my_quiz.yaml
# or
QUIZGEN_ALLOW_GENERATOR=1 quizgen --yaml my_quiz.yaml
```

If you need dynamic question generation with untrusted inputs, consider writing a proper `Question` subclass instead, which provides better control and validation.

### LaTeX `-shell-escape` Warning

When using `--latex`, QuizGenerator invokes `latexmk -shell-escape` to compile PDFs. This allows LaTeX to execute external commands (for example, via `\write18`). If your question content includes raw LaTeX (e.g., from custom question types or untrusted YAML sources), this can be a command‑execution vector.

Guidance:
- Only use `--latex` with trusted question sources.
- Prefer Typst (default) when possible.
- If you need LaTeX but want to reduce risk, avoid raw LaTeX content and keep custom questions constrained to ContentAST elements.

## Project Structure

```
QuizGenerator/
├── QuizGenerator/           # Main package
│   ├── question.py         # Question base classes and registry
│   ├── quiz.py            # Quiz generation logic
│   ├── contentast.py      # Content AST for cross-format rendering
│   ├── premade_questions/ # Built-in question library
│   └── ...               # Question types and rendering utilities
├── example_files/        # Example quiz configurations
├── documentation/        # User guides
├── lms_interface/        # Canvas LMS integration
└── quizgen             # CLI entry point
```

## Contributing

Contributions welcome! Areas of interest:
- New question types
- Additional LMS integrations
- Documentation improvements
- Bug fixes

## License

GNU General Public License v3.0 or later (GPLv3+) - see LICENSE file for details

## Citation

If you use QuizGenerator in academic work, please cite:

```
@software{quizgenerator,
  author = {Ogden, Sam},
  title = {QuizGenerator: Automated Quiz Generation for Education},
  year = {2024},
  url = {https://github.com/OtterDen-Lab/QuizGenerator}
}
```

## Support

- Issues: https://github.com/OtterDen-Lab/QuizGenerator/issues
- Documentation: https://github.com/OtterDen-Lab/QuizGenerator/tree/main/documentation

---

**Note**: This tool is designed for educational use. Ensure compliance with your institution's academic integrity policies when using automated quiz generation.
