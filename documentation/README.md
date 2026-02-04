# QuizGeneration Documentation

This directory contains documentation for the QR code-based quiz grading system.

## Documentation Files

### User Guides

#### [GRADING_GUIDE.md](GRADING_GUIDE.md)
Complete guide for instructors and graders using the QR code system for grading printed exams.

**Topics covered:**
- QR code system overview
- Setup and installation of grading tools
- Encryption key management
- Grading workflows (CLI-based)
- Scanning QR codes from images
- Troubleshooting common issues
- Security considerations
- Question versioning

**Audience:** Instructors, TAs, and graders using printed exam PDFs

---

#### [WEB_UI_INTEGRATION.md](WEB_UI_INTEGRATION.md)
Technical guide for developers integrating QuizGeneration with web-based grading interfaces.

**Topics covered:**
- Installing QuizGeneration as a library
- Python API for answer regeneration
- Web framework examples (Flask, Django)
- QR code data format specification
- Answer object types and formats
- Error handling
- Performance considerations
- Example code and complete workflows

**Audience:** Web developers building grading interfaces

---

### Developer Documentation

#### [PARAMETER_STANDARDS.md](PARAMETER_STANDARDS.md)
Standards and conventions for question parameter naming and design.

**Topics covered:**
- Parameter naming conventions
- Best practices for question design
- Consistency guidelines

**Audience:** Question developers

---

#### [LESSONS_LEARNED-adding_questions.md](LESSONS_LEARNED-adding_questions.md)
Accumulated knowledge and gotchas from developing new question types.

**Topics covered:**
- Common pitfalls when creating questions
- Tips and tricks for question development
- Solutions to recurring problems
- Implementation patterns

**Audience:** Question developers

---

#### [claude_todo.md](claude_todo.md)
Project todo list and task tracking.

**Audience:** Project maintainers

---

## Quick Start

### For Graders (Using CLI)
See [GRADING_GUIDE.md](GRADING_GUIDE.md)

```bash
# Install grading dependencies
uv sync --extra grading

# Set encryption key
export QUIZ_ENCRYPTION_KEY="your-key-here"

# Scan QR code from image
python grade_from_qr.py --image scanned_exam.jpg
```

### For Web Developers (Using API)
See [WEB_UI_INTEGRATION.md](WEB_UI_INTEGRATION.md)

```python
# Install as library
pip install -e /path/to/QuizGeneration

# Use in your code
from grade_from_qr import regenerate_from_metadata

result = regenerate_from_metadata(
    question_type="VirtualAddressParts",
    seed=12345,
    version="1.0",
    points=2.0
)
```

---

## System Overview

The QR code grading system allows automated answer regeneration for randomized exams:

1. **PDF Generation**: Each question gets a QR code with encrypted metadata (question type, seed, version)
2. **Exam Administration**: Students complete printed exams with QR codes
3. **Grading**: Scan QR codes to regenerate correct answers without managing multiple answer keys

**Key Benefits:**
- No need to track 90+ different answer keys
- Deterministic answer regeneration from compact metadata
- Works offline (no network required)
- Secure encoding prevents answer extraction by students

---

## Project Structure

```
QuizGeneration/
├── documentation/           # This directory
│   ├── README.md           # This file
│   ├── GRADING_GUIDE.md    # CLI grading guide
│   └── WEB_UI_INTEGRATION.md  # Web API guide
├── examples/
│   └── web_ui_integration_example.py  # Working code examples
├── grade_from_qr.py        # Main grading module (CLI + library)
├── QuizGenerator/
│   ├── qrcode_generator.py # QR code generation/encryption
│   ├── question.py         # Question classes and registry
│   └── ...
└── quizgen                # PDF generation entry point
```

---

## Support

For issues or questions:
- Check the troubleshooting sections in the guides
- Review `examples/web_ui_integration_example.py` for working code
- Verify environment setup (encryption key, dependencies)

---

*Last updated: 2025-10-09*
