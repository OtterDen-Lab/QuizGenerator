# Web UI Integration Guide

This guide shows how to integrate QuizGeneration with your web grading interface to regenerate answers from QR codes.

## Overview

Your web grading UI will:
1. Capture an image of a QR code from a scanned exam
2. Decode the QR code to extract question metadata
3. Call this module's API to regenerate the correct answers
4. Display answers in your grading interface

## Installation

### Option 1: Install via pip (if published to PyPI)
```bash
pip install QuizGenerator
```

### Option 2: Install from local directory
```bash
# Clone or place this repo next to your web UI project
cd /path/to/your/web/ui/parent/directory
git clone <this-repo-url> QuizGeneration

# Install in editable mode
pip install -e ./QuizGeneration
```

### Option 3: Add as git submodule
```bash
cd /path/to/your/web/ui
git submodule add <this-repo-url> vendor/QuizGeneration
pip install -e vendor/QuizGeneration
```

## Setup

### 1. Install Dependencies

```bash
# Core dependencies (required)
pip install QuizGenerator

# Optional: QR scanning for CLI tool
pip install "QuizGenerator[grading]"  # Includes pyzbar and pillow
```

### 2. Set Encryption Key

The same encryption key used to generate PDFs must be available when grading:

```python
import os

# Option A: Set in code (not recommended for production)
os.environ['QUIZ_ENCRYPTION_KEY'] = 'your-key-here'

# Option B: Load from .env file (recommended)
from dotenv import load_dotenv
load_dotenv()  # Reads QUIZ_ENCRYPTION_KEY from .env

# Option C: Set in deployment environment (best for production)
# Set QUIZ_ENCRYPTION_KEY in your server's environment variables
```

## Web UI Integration

### Simple API (Recommended)

If your web UI already decodes QR codes and extracts the metadata, use this simple API:

```python
from grade_from_qr import regenerate_from_metadata

def grade_question(question_type: str, seed: int, version: str, points: float):
    """
    Your web endpoint that receives decoded QR metadata.

    Example POST request:
    {
        "question_type": "VirtualAddressParts",
        "seed": 12345,
        "version": "1.0",
        "points": 2.0
    }
    """
    try:
        result = regenerate_from_metadata(question_type, seed, version, points)

        # Extract answers for display
        answers = []
        for key, answer_obj in result['answer_objects'].items():
            answer_dict = {
                "key": key,
                "value": answer_obj.value
            }

            # Include tolerance for numerical answers
            if hasattr(answer_obj, 'tolerance') and answer_obj.tolerance:
                answer_dict['tolerance'] = answer_obj.tolerance

            answers.append(answer_dict)

        return {
            "success": True,
            "answers": answers
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
```

### Full Flow (with QR Decoding)

If you want this module to handle QR decoding:

```python
import json
from grade_from_qr import scan_qr_from_image, parse_qr_data, regenerate_from_metadata
from QuizGenerator.qrcode_generator import QuestionQRCode

def grade_from_qr_image(image_path: str):
    """
    Full flow: QR image -> decoded metadata -> regenerated answers.

    Args:
        image_path: Path to image file containing QR code
    """
    # Step 1: Scan QR code from image
    qr_codes = scan_qr_from_image(image_path)
    if not qr_codes:
        return {"error": "No QR code found"}

    # Step 2: Parse QR JSON
    qr_data = parse_qr_data(qr_codes[0])
    # Returns: {"q": 1, "pts": 2.0, "s": "encrypted_string"}

    # Step 3: Decrypt metadata
    encrypted = qr_data.get('s')
    if not encrypted:
        return {"error": "QR code missing regeneration data"}

    metadata = QuestionQRCode.decrypt_question_data(encrypted)
    # Returns: {"question_type": "...", "seed": 123, "version": "1.0"}

    # Step 4: Regenerate answers
    result = regenerate_from_metadata(
        question_type=metadata['question_type'],
        seed=metadata['seed'],
        version=metadata['version'],
        points=qr_data['pts']
    )

    return {
        "question_number": qr_data['q'],
        "answers": format_answers(result['answer_objects'])
    }

def format_answers(answer_objects):
    """Format Answer objects for web display."""
    return [
        {
            "key": key,
            "value": ans.value,
            "tolerance": getattr(ans, 'tolerance', None)
        }
        for key, ans in answer_objects.items()
    ]
```

## Example: Flask Web API

```python
from flask import Flask, request, jsonify
from grade_from_qr import regenerate_from_metadata
from QuizGenerator.qrcode_generator import QuestionQRCode
import json

app = Flask(__name__)

@app.route('/api/grade/regenerate', methods=['POST'])
def regenerate_answer():
    """
    API endpoint to regenerate answers from QR metadata.

    Request body:
    {
        "qr_data": {
            "q": 1,
            "pts": 2.0,
            "s": "Yx8CBgc5DjAUVDdQCTcXNUcCA0hDalFFRQp0G0o="
        }
    }
    """
    try:
        qr_data = request.json['qr_data']

        # Decrypt metadata
        encrypted = qr_data['s']
        metadata = QuestionQRCode.decrypt_question_data(encrypted)

        # Regenerate answers
        result = regenerate_from_metadata(
            question_type=metadata['question_type'],
            seed=metadata['seed'],
            version=metadata['version'],
            points=qr_data['pts']
        )

        # Format response
        answers = [
            {
                "key": k,
                "value": str(v.value),
                "tolerance": getattr(v, 'tolerance', None)
            }
            for k, v in result['answer_objects'].items()
        ]

        return jsonify({
            "success": True,
            "question_number": qr_data['q'],
            "points": qr_data['pts'],
            "question_type": metadata['question_type'],
            "answers": answers
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True)
```

## Example: Django View

```python
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from grade_from_qr import regenerate_from_metadata
from QuizGenerator.qrcode_generator import QuestionQRCode
import json

@csrf_exempt
def regenerate_answer_view(request):
    """
    Django view to regenerate answers from QR metadata.
    """
    if request.method != 'POST':
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        data = json.loads(request.body)
        qr_data = data['qr_data']

        # Decrypt and regenerate
        encrypted = qr_data['s']
        metadata = QuestionQRCode.decrypt_question_data(encrypted)

        result = regenerate_from_metadata(
            question_type=metadata['question_type'],
            seed=metadata['seed'],
            version=metadata['version'],
            points=qr_data['pts']
        )

        # Format answers
        answers = [
            {
                "key": k,
                "value": str(v.value),
                "tolerance": getattr(v, 'tolerance', None)
            }
            for k, v in result['answer_objects'].items()
        ]

        return JsonResponse({
            "success": True,
            "question_number": qr_data['q'],
            "answers": answers
        })

    except Exception as e:
        return JsonResponse({
            "success": False,
            "error": str(e)
        }, status=400)
```

## QR Code Format

The QR codes contain JSON with this structure:

```json
{
    "q": 1,                                    // Question number
    "pts": 2.0,                               // Point value
    "s": "Yx8CBgc5DjAUVDdQCTcXNUcCA0hDalFFRQp0G0o="  // Encrypted metadata
}
```

The encrypted `"s"` field decodes to:
```json
{
    "question_type": "VirtualAddressParts",   // Question class name
    "seed": 12345,                            // Random seed
    "version": "1.0"                          // Question version
}
```

## Answer Object Format

The `regenerate_from_metadata()` function returns:

```python
{
    "question_type": "VirtualAddressParts",
    "seed": 12345,
    "version": "1.0",
    "points": 2.0,
    "answers": {                              # Canvas-formatted answers
        "kind": "numerical_answer",
        "data": [...]
    },
    "answer_objects": {                       # Raw Answer objects
        "page_offset": Answer(value=28, tolerance=None),
        "vpn": Answer(value=45, tolerance=None)
    }
}
```

### Answer Object Types

Common answer object types and their attributes:

1. **Numerical Answer**
   ```python
   Answer(value=42, tolerance=0.1)
   # Student answer must be within [41.9, 42.1]
   ```

2. **String Answer**
   ```python
   Answer(value="0xDEADBEEF", tolerance=None)
   # Exact string match required
   ```

3. **Multiple Choice** (via Canvas format)
   ```python
   # Check result['answers']['data'] for Canvas-formatted choices
   ```

## Error Handling

```python
from grade_from_qr import regenerate_from_metadata

try:
    result = regenerate_from_metadata("UnknownQuestion", 12345, "1.0")
except ValueError as e:
    # Question type not found or regeneration failed
    print(f"Error: {e}")
except Exception as e:
    # Other errors (missing key, invalid data, etc.)
    print(f"Unexpected error: {e}")
```

Common errors:
- `ValueError`: Question type not registered or regeneration failed
- `ImportError`: Missing dependencies (encryption libraries)
- `KeyError`: Invalid QR data format

## Testing

Test the integration with your web UI:

```python
import os
os.environ['QUIZ_ENCRYPTION_KEY'] = 'your-test-key'

from grade_from_qr import regenerate_from_metadata

# Test regeneration
result = regenerate_from_metadata(
    question_type="VirtualAddressParts",
    seed=12345,
    version="1.0",
    points=2.0
)

print(f"Generated {len(result['answer_objects'])} answers:")
for key, answer in result['answer_objects'].items():
    print(f"  {key}: {answer.value}")
```

## Security Notes

1. **Encryption Key**: Keep your `QUIZ_ENCRYPTION_KEY` secret and secure
2. **Key Rotation**: If you change the key, old exams cannot be graded
3. **Version Checking**: Always check the version number to detect incompatible question changes
4. **Obfuscation**: The encoding uses XOR obfuscation, not cryptographically secure encryption

## Performance

- Answer regeneration is fast (~10-50ms per question)
- QR decoding depends on your chosen library
- Consider caching regenerated answers by (question_type, seed, version) tuple
- No network calls required - all computation is local

## Example Projects

See `web_ui_integration_example.py` for a complete working example showing:
- Full QR scan flow
- Direct metadata API
- Answer formatting for web display

## Troubleshooting

### ImportError: No module named 'QuizGenerator'
```bash
pip install -e /path/to/QuizGeneration
```

### Missing encryption key
```bash
export QUIZ_ENCRYPTION_KEY="your-key-here"
```

### Question type not found
- Ensure all question classes are loaded (import the premade_questions modules)
- Check that the question class still exists in the codebase

### Version mismatch
- Question code changed between PDF generation and grading
- Consider regenerating PDFs with updated question versions

---

**For more examples, see**: `web_ui_integration_example.py`
