# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based teaching tools project focused on quiz generation and Canvas LMS integration. The main entry point is `generate_quiz.py`, which generates quizzes in both PDF format and Canvas variations based on YAML configuration files.

## Architecture

### Core Modules
- **QuizGenerator/**: Main quiz generation engine
  - `quiz.py`: Core Quiz class that loads from YAML and generates output
  - `question.py`: Question classes and registry system for different question types
  - `premade_questions/`: Collection of specialized question generators (memory, processes, persistence, etc.)
  - `misc.py`: Utilities for output formats and content handling

- **lms_interface/**: Canvas LMS integration (submodule)
  - `canvas_interface.py`: CanvasInterface and CanvasCourse classes for API interaction
  - `classes.py`: Data models for LMS objects

### Question System
The project uses a plugin-based question system where question types are defined as classes in `premade_questions/` and registered dynamically. Questions are defined in YAML files that specify the class name and parameters.

### Configuration Files
Quiz configurations are stored in `example_files/` as YAML files. The default configuration is `exam_generation.yaml`, which defines question types, point values, and organization.

## Development Commands

### Environment Setup
```bash
# The project uses uv for dependency management
uv sync
source .venv/bin/activate
```

### Running the Main Script
```bash
# Generate PDFs only
python generate_quiz.py --num_pdfs 3

# Generate Canvas quizzes (requires course_id)
python generate_quiz.py --num_canvas 5 --course_id 12345

# Use production Canvas instance
python generate_quiz.py --prod --num_canvas 2 --course_id 12345

# Custom quiz configuration
python generate_quiz.py --quiz_yaml example_files/custom_quiz.yaml --num_pdfs 2
```

### Quality Tools
The project includes development dependencies for code quality:
```bash
# Code formatting
black .

# Linting
flake8

# Type checking
mypy QuizGenerator/

# Testing
pytest
```

### LaTeX Dependencies
PDF generation requires LaTeX with `latexmk`. Generated PDFs are output to the `out/` directory.

## Key Files and Patterns

- Quiz configurations follow the pattern in `example_files/exam_generation.yaml`
- New question types should extend base classes in `question.py` and be placed in `premade_questions/`
- The project uses logging extensively - check `teachingtools.log` for detailed output
- Canvas integration requires environment variables for API keys (see `.env` symlink)

## Testing Notes

The project includes a test mode accessible via `python generate_quiz.py TEST`.