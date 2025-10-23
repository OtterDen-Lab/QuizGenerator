#!/usr/bin/env python
"""
QR Code-based Grading Utility

This script scans QR codes from quiz PDFs to regenerate question answers
for grading. It supports both scanning from image files and interactive
scanning from webcam/scanner.

CLI Usage:
    # Scan a single QR code image
    python grade_from_qr.py --image qr_code.png

    # Scan QR codes from a scanned exam page
    python grade_from_qr.py --image exam_page.jpg --all

API Usage (recommended for web UIs):
    from grade_from_qr import regenerate_from_encrypted

    # Parse QR code JSON
    qr_data = json.loads(qr_string)

    # Regenerate answers from encrypted data (one function call!)
    result = regenerate_from_encrypted(
        encrypted_data=qr_data['s'],
        points=qr_data['pts']
    )

    # Display HTML answer key to grader
    print(result['answer_key_html'])

The QR codes contain encrypted question metadata that allows regenerating
the exact question and answer without needing the original exam file.
"""

import argparse
import json
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Quiz generator imports (always available)
from QuizGenerator.qrcode_generator import QuestionQRCode
from QuizGenerator.question import QuestionRegistry

# QR code reading (optional - only needed for CLI usage with --image)
# Your web UI should use its own QR decoding library
try:
    from pyzbar import pyzbar
    from PIL import Image
    HAS_IMAGE_SUPPORT = True
except ImportError:
    HAS_IMAGE_SUPPORT = False
    # Don't fail immediately - only fail if user tries to use --image flag

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
log = logging.getLogger(__name__)


def scan_qr_from_image(image_path: str) -> List[str]:
    """
    Scan all QR codes from an image file.

    Args:
        image_path: Path to image file containing QR code(s)

    Returns:
        List of decoded QR code data strings

    Raises:
        ImportError: If pyzbar/PIL are not installed
    """
    if not HAS_IMAGE_SUPPORT:
        raise ImportError(
            "Image support not available. Install with: pip install pyzbar pillow"
        )

    try:
        img = Image.open(image_path)
        decoded_objects = pyzbar.decode(img)

        if not decoded_objects:
            log.warning(f"No QR codes found in {image_path}")
            return []

        qr_data = [obj.data.decode('utf-8') for obj in decoded_objects]
        log.info(f"Found {len(qr_data)} QR code(s) in {image_path}")

        return qr_data

    except Exception as e:
        log.error(f"Failed to read image {image_path}: {e}")
        return []


def parse_qr_data(qr_string: str) -> Dict[str, Any]:
    """
    Parse QR code JSON data.

    Args:
        qr_string: JSON string from QR code

    Returns:
        Dictionary with question metadata
        {
            "q": question_number,
            "pts": points_value,
            "s": encrypted_seed_data (optional)
        }
    """
    try:
        data = json.loads(qr_string)
        log.debug(f"Parsed QR data: {data}")
        return data
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse QR code JSON: {e}")
        return {}


def regenerate_question_answer(qr_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Regenerate question and extract answer using QR code metadata.

    Args:
        qr_data: Parsed QR code data dictionary

    Returns:
        Dictionary with question info and answers, or None if regeneration fails
        {
            "question_number": int,
            "points": float,
            "question_type": str,
            "seed": int,
            "version": str,
            "answers": dict
        }
    """
    question_num = qr_data.get('q')
    points = qr_data.get('pts')

    if question_num is None or points is None:
        log.error("QR code missing required fields 'q' or 'pts'")
        return None

    result = {
        "question_number": question_num,
        "points": points
    }

    # Check if encrypted regeneration data is present
    encrypted_data = qr_data.get('s')
    if not encrypted_data:
        log.warning(f"Question {question_num}: No regeneration data in QR code")
        log.warning("  This question cannot be automatically regenerated.")
        log.warning("  (QR codes generated before encryption feature was added)")
        return result

    try:
        # Decrypt the regeneration data
        regen_data = QuestionQRCode.decrypt_question_data(encrypted_data)

        question_type = regen_data['question_type']
        seed = regen_data['seed']
        version = regen_data['version']
        config = regen_data.get('config', {})

        result['question_type'] = question_type
        result['seed'] = seed
        result['version'] = version
        if config:
            result['config'] = config

        log.info(f"Question {question_num}: {question_type} (seed={seed}, version={version})")
        if config:
            log.debug(f"  Config params: {config}")

        # Regenerate the question using the registry, passing through config params
        question = QuestionRegistry.create(
            question_type,
            name=f"Q{question_num}",
            points_value=points,
            **config
        )

        # Generate question with the specific seed
        question_ast = question.get_question(rng_seed=seed)

        # Extract answers
        answer_kind, canvas_answers = question.get_answers()

        result['answers'] = {
            'kind': answer_kind.value,
            'data': canvas_answers
        }

        # Also store the raw answer objects for easier access
        result['answer_objects'] = question.answers

        # Generate HTML answer key for grading
        question_html = question_ast.body.render("html", show_answers=True)
        result['answer_key_html'] = question_html

        log.info(f"  Successfully regenerated question with {len(canvas_answers)} answer(s)")

        return result

    except Exception as e:
        log.error(f"Failed to regenerate question {question_num}: {e}")
        import traceback
        log.debug(traceback.format_exc())
        return result


def regenerate_from_encrypted(encrypted_data: str, points: float = 1.0) -> Dict[str, Any]:
    """
    Regenerate question answers from encrypted QR code data (RECOMMENDED API).

    This is the simplest function for web UI integration - just pass the encrypted
    string from the QR code and get back the complete answer key.

    Args:
        encrypted_data: The encrypted 's' field from the QR code JSON
        points: Point value for the question (default: 1.0)

    Returns:
        Dictionary with regenerated answers:
        {
            "question_type": str,
            "seed": int,
            "version": str,
            "points": float,
            "kwargs": dict,  # Question-specific config params (if any)
            "answers": dict,  # Canvas-formatted answers
            "answer_objects": dict,  # Raw Answer objects with values/tolerances
            "answer_key_html": str  # HTML rendering of question with answers shown
        }

    Raises:
        ValueError: If decryption fails or question regeneration fails

    Example:
        >>> # Your web UI scans QR code and gets JSON: {"q": 1, "pts": 5, "s": "gAAAAAB..."}
        >>> encrypted_string = qr_json['s']
        >>> result = regenerate_from_encrypted(encrypted_string, points=qr_json['pts'])
        >>> print(result['answer_key_html'])  # Display to grader!
    """
    # Decrypt the data
    decrypted = QuestionQRCode.decrypt_question_data(encrypted_data)

    # Extract fields
    question_type = decrypted['question_type']
    seed = decrypted['seed']
    version = decrypted['version']
    kwargs = decrypted.get('config', {})

    # Use the existing regeneration logic
    return regenerate_from_metadata(question_type, seed, version, points, kwargs)


def regenerate_from_metadata(question_type: str, seed: int, version: str,
                            points: float = 1.0, kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Regenerate question answers from explicit metadata fields.

    This is a lower-level function. Most users should use regenerate_from_encrypted() instead.

    Args:
        question_type: Question class name (e.g., "VirtualAddressParts")
        seed: Random seed used to generate the question
        version: Question version string (e.g., "1.0")
        points: Point value for the question (default: 1.0)
        kwargs: Optional dictionary of question-specific configuration parameters
                (e.g., {"num_bits_va": 32, "max_value": 100})

    Returns:
        Dictionary with regenerated answers (same format as regenerate_from_encrypted)

    Raises:
        ValueError: If question type is not registered or regeneration fails
    """
    if kwargs is None:
        kwargs = {}

    try:
        log.info(f"Regenerating: {question_type} (seed={seed}, version={version})")
        if kwargs:
            log.debug(f"  Config params: {kwargs}")

        # Create question instance from registry, passing through kwargs
        question = QuestionRegistry.create(
            question_type,
            name=f"Q_{question_type}_{seed}",
            points_value=points,
            **kwargs
        )

        # Generate question with the specific seed
        question_ast = question.get_question(rng_seed=seed)

        # Extract answers
        answer_kind, canvas_answers = question.get_answers()

        # Generate HTML answer key for grading
        question_html = question_ast.body.render("html", show_answers=True)

        result = {
            "question_type": question_type,
            "seed": seed,
            "version": version,
            "points": points,
            "answers": {
                "kind": answer_kind.value,
                "data": canvas_answers
            },
            "answer_objects": question.answers,
            "answer_key_html": question_html
        }

        # Include kwargs in result if provided
        if kwargs:
            result["kwargs"] = kwargs

        log.info(f"  Successfully regenerated with {len(canvas_answers)} answer(s)")
        return result

    except Exception as e:
        log.error(f"Failed to regenerate question: {e}")
        import traceback
        log.debug(traceback.format_exc())
        raise ValueError(f"Failed to regenerate question {question_type}: {e}")


def display_answer_summary(question_data: Dict[str, Any]) -> None:
    """
    Display a formatted summary of the question and its answer(s).

    Args:
        question_data: Question data dictionary from regenerate_question_answer
    """
    print("\n" + "="*60)
    print(f"Question {question_data['question_number']}: {question_data.get('points', '?')} points")

    if 'question_type' in question_data:
        print(f"Type: {question_data['question_type']}")
        print(f"Seed: {question_data['seed']}")
        print(f"Version: {question_data['version']}")

    if 'answer_objects' in question_data:
        print("\nANSWERS:")
        for key, answer_obj in question_data['answer_objects'].items():
            print(f"  {key}: {answer_obj.value}")
            if hasattr(answer_obj, 'tolerance') and answer_obj.tolerance:
                print(f"    (tolerance: ±{answer_obj.tolerance})")
    elif 'answers' in question_data:
        print("\nANSWERS (raw Canvas format):")
        print(f"  Type: {question_data['answers']['kind']}")
        for ans in question_data['answers']['data']:
            print(f"  - {ans}")
    else:
        print("\n(No regeneration data available)")

    if 'answer_key_html' in question_data:
        print("\nHTML answer key available in result['answer_key_html']")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Scan QR codes from quiz PDFs to regenerate answers for grading"
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to image file containing QR code(s)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all QR codes found in the image (default: only first one)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Save results to JSON file'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose debug logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.image:
        parser.print_help()
        print("\nERROR: --image is required")
        sys.exit(1)

    # Scan QR codes from image
    qr_codes = scan_qr_from_image(args.image)

    if not qr_codes:
        print("No QR codes found in image")
        sys.exit(1)

    # Process QR codes
    if not args.all:
        qr_codes = qr_codes[:1]
        log.info("Processing only the first QR code (use --all to process all)")

    results = []

    for qr_string in qr_codes:
        # Parse QR data
        qr_data = parse_qr_data(qr_string)

        if not qr_data:
            continue

        # Regenerate question and answer
        question_data = regenerate_question_answer(qr_data)

        if question_data:
            results.append(question_data)
            display_answer_summary(question_data)

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        log.info(f"Results saved to {output_path}")

    if not results:
        print("\nNo questions could be regenerated from the QR codes.")
        sys.exit(1)

    print(f"\nSuccessfully regenerated {len(results)} question(s)")


if __name__ == '__main__':
    main()
