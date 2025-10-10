# QR Code-Based Grading Guide

This guide explains how to use the QR code system for automated answer regeneration when grading randomized exams.

## Overview

Each question in generated PDF exams includes a QR code containing encrypted metadata. This metadata allows you to regenerate the exact question and its correct answer without needing the original exam file or managing 90+ different answer keys.

## How It Works

1. **Question Generation**: When a quiz is generated, each question is assigned:
   - A unique random seed
   - A question type/class name
   - A version number

2. **QR Code Embedding**: This metadata is encrypted and embedded in a QR code on each question in the PDF

3. **Scanning & Regeneration**: During grading, you scan the QR code to decrypt the metadata and regenerate the exact question with the correct answer

## Setup

### Install Grading Dependencies

The grading tools require additional packages for QR code scanning:

```bash
# Using uv (recommended)
uv sync --extra grading

# Or using pip
pip install pyzbar pillow
```

**Note**: On some systems, `pyzbar` requires the ZBar library to be installed:

- **macOS**: `brew install zbar`
- **Ubuntu/Debian**: `sudo apt-get install libzbar0`
- **Windows**: Download from http://zbar.sourceforge.net/

### Set Encryption Key (REQUIRED)

QR codes use encrypted data to prevent tampering. You MUST set an encryption key as an environment variable:

```bash
# Generate a new key (do this ONCE and save it!)
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Set the environment variable (add to your .bashrc or .zshrc)
export QUIZ_ENCRYPTION_KEY="your-generated-key-here"
```

**IMPORTANT**:
- Use the SAME key for generating exams and grading them
- Store this key securely - without it, you cannot grade QR code-based exams
- Never commit the key to version control

## Grading Workflow

### Method 1: Scan Individual QR Codes

If you have a smartphone or QR scanner app, you can extract QR codes individually:

```bash
# Scan a single QR code image
python grade_from_qr.py --image qr_code_q3.png
```

Output:
```
============================================================
Question 3: 5.0 points
Type: VectorDotProduct
Seed: 12345
Version: 1.0

ANSWERS:
  result: 42
    (tolerance: ±0.1)
============================================================
```

### Method 2: Scan Entire Exam Page

Scan a full exam page and extract all QR codes at once:

```bash
# Process all QR codes in the scanned page
python grade_from_qr.py --image scanned_exam_page1.jpg --all
```

This will regenerate answers for all questions on that page.

### Method 3: Batch Processing with JSON Output

For automated grading systems, save results to JSON:

```bash
python grade_from_qr.py --image exam_page1.jpg --all --output answers_page1.json
```

The JSON output contains structured answer data that can be processed programmatically:

```json
[
  {
    "question_number": 1,
    "points": 5.0,
    "question_type": "MatrixDeterminant",
    "seed": 67890,
    "version": "1.0",
    "answers": {
      "kind": "numerical_answer",
      "data": [{"text": "42", "exact": 42, "margin": 0.1}]
    },
    "answer_objects": {
      "determinant": "Answer(value=42, tolerance=0.1)"
    }
  }
]
```

## Troubleshooting

### QR Code Not Scanning

**Problem**: `No QR codes found in image`

**Solutions**:
- Ensure image quality is sufficient (at least 200 DPI recommended)
- Crop the image to show just the QR code area
- Adjust lighting/contrast in the scanned image
- Try a different scanning app or method

### Decryption Errors

**Problem**: `Failed to decrypt QR code data`

**Solutions**:
- Verify `QUIZ_ENCRYPTION_KEY` environment variable is set correctly
- Ensure you're using the SAME key that was used to generate the exam
- Check that the QR code contains encrypted data (older exams may not have this)

### Question Regeneration Failures

**Problem**: `Unknown question type: SomeQuestion`

**Solutions**:
- Ensure the question type still exists in the codebase
- Check that you've loaded the correct premade questions
- Verify the question version matches (version incompatibilities may cause issues)

### Legacy QR Codes

**Problem**: `No regeneration data in QR code`

This occurs with QR codes generated before the encryption feature was added. These QR codes only contain:
- Question number
- Point value

They cannot be automatically regenerated. You'll need to refer to the original answer key.

## Security Considerations

### Why Encryption?

The QR codes use Fernet symmetric encryption to:
1. **Prevent answer extraction**: Students can't decode QR codes without the key
2. **Ensure authenticity**: Tampering with QR codes will cause decryption to fail
3. **Enable versioning**: Version numbers detect when question logic has changed

### Key Management Best Practices

1. **Generate once**: Create a single key and use it for all exams in a course
2. **Store securely**: Keep the key in a password manager or secure environment file
3. **Backup**: Store a backup of the key in a secure location
4. **Rotate carefully**: If you change the key, old exams cannot be graded with new key

### Environment Setup Example

Create a `.env` file (DO NOT commit to git):

```bash
# .env (add to .gitignore!)
QUIZ_ENCRYPTION_KEY=your-generated-key-here
```

Load it in your shell:

```bash
# Add to .bashrc or .zshrc
export $(cat .env | xargs)
```

## Advanced Usage

### Scanning from Webcam (Future Feature)

Interactive webcam scanning is planned for future releases:

```bash
# Not yet implemented
python grade_from_qr.py --interactive
```

### Batch Answer Key Generation

Generate answer keys for all exams in a directory:

```bash
# Pseudocode - implement as needed
for exam in exams/*.pdf; do
    # Extract QR codes from PDF
    # Scan each page
    python grade_from_qr.py --image "$exam" --all --output "answers_$(basename $exam).json"
done
```

## Question Versioning

Each question class has a `VERSION` attribute (e.g., `VERSION = "1.0"`). Increment this when you change:

- Random number generation logic
- Question calculation methods
- Answer generation algorithms

**DO NOT increment for**:
- Cosmetic changes (formatting, wording)
- Bug fixes that don't affect answers
- Changes to explanations only

Versioning ensures that regenerated answers match the questions that were actually printed.

## Example: Complete Grading Workflow

1. **Generate exams with QR codes**:
   ```bash
   export QUIZ_ENCRYPTION_KEY="your-key-here"
   python generate_quiz.py --num_pdfs 90 --quiz_yaml exam_config.yaml
   ```

2. **Students complete exams** (printed PDFs with QR codes)

3. **Scan completed exam**:
   ```bash
   # Scan each page to an image (use phone camera or scanner)
   # Save as exam_student123_page1.jpg, etc.
   ```

4. **Extract answers**:
   ```bash
   python grade_from_qr.py --image exam_student123_page1.jpg --all
   ```

5. **Compare student answers to regenerated answers**

6. **Grade and provide feedback**

## FAQ

**Q: Can students decode the QR codes to get answers?**
A: No. The data is encrypted with a key only you possess.

**Q: What if I lose the encryption key?**
A: You won't be able to regenerate answers from QR codes. Keep backups!

**Q: Do I need to save 90 different answer keys?**
A: No! That's the whole point - the QR code IS the answer key.

**Q: What if question code changes between generating and grading?**
A: Version numbers detect this. If versions mismatch, you'll get a warning.

**Q: Can I use this with Canvas/online exams?**
A: No, QR codes are only embedded in PDF output for printed exams.

## Support

For issues or questions:
- Check the logs with `--verbose` flag for debugging
- Review the question version numbers if regeneration seems incorrect
- Verify your environment setup and encryption key

---

*Last updated: 2025-10-09*
