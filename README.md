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

### System Requirements

- Python 3.12+
- LaTeX distribution with `latexmk` (for PDF generation)
- Optional: [Typst](https://typst.app/) (alternative to LaTeX)

### Optional Dependencies

```bash
# For QR code grading support
pip install QuizGenerator[grading]
```

## Quick Start

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

### 2. Generate PDFs

```bash
python -m generate_quiz --quiz_yaml my_quiz.yaml --num_pdfs 3
```

PDFs will be created in the `out/` directory.

### 3. Upload to Canvas

```bash
# Set up Canvas credentials in ~/.env first:
# CANVAS_API_URL=https://canvas.instructure.com
# CANVAS_API_KEY=your_api_key_here

python -m generate_quiz \
  --quiz_yaml my_quiz.yaml \
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
- `FromGenerator` - Programmatically generated questions

## Documentation

- [Getting Started Guide](documentation/getting_started.md) (coming soon)
- [Custom Questions Guide](documentation/custom_questions.md)
- [YAML Configuration Reference](documentation/yaml_config_guide.md) (coming soon)
- [PyPI Release Plan](documentation/pypi_release_plan.md)

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
python -m generate_quiz --prod --num_canvas 5 --course_id 12345
```

## Advanced Features

### Typst Support

Use Typst instead of LaTeX for faster compilation:

```bash
python -m generate_quiz --typst --num_pdfs 3
```

### Deterministic Generation

Use seeds for reproducible quizzes:

```bash
python -m generate_quiz --seed 42 --num_pdfs 3
```

### QR Code Regeneration

Each generated exam includes a QR code that stores:
- Question types and parameters
- Random seed
- Version information

Use the grading tools to scan QR codes and regenerate exact exam versions.

## Project Structure

```
QuizGenerator/
├── QuizGenerator/           # Main package
│   ├── question.py         # Question base classes and registry
│   ├── quiz.py            # Quiz generation logic
│   ├── contentast.py      # Content AST for cross-format rendering
│   ├── premade_questions/ # Built-in question library
│   └── canvas/           # Canvas LMS integration
├── example_files/        # Example quiz configurations
├── documentation/        # User guides
└── generate_quiz.py     # CLI entry point
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