#!/usr/bin/env python
"""
Example: Integrating QuizGeneration with a Web Grading UI

This shows how your web UI should import and use the grading module
to regenerate answers from QR code data.
"""

import os
from grade_from_qr import regenerate_from_metadata
from QuizGenerator.qrcode_generator import QuestionQRCode

# ============================================================================
# SETUP: Set your encryption key (same one used to generate PDFs)
# ============================================================================

# Option 1: Set environment variable
# os.environ['QUIZ_ENCRYPTION_KEY'] = 'your-key-here'

# Option 2: Load from .env file (recommended for web apps)
# from dotenv import load_dotenv
# load_dotenv()


# ============================================================================
# WEB UI INTEGRATION FLOW
# ============================================================================

def handle_qr_scan(qr_image_data):
    """
    Example: Your web UI receives a QR code image from the grading interface.

    Args:
        qr_image_data: Image data (bytes, PIL Image, or file path)

    Returns:
        dict: Answer data to display in grading UI
    """

    # Step 1: Your web UI decodes the QR image to get the JSON string
    # (You'll use your own QR decoding library - pyzbar, zxing, etc.)
    import json
    from PIL import Image
    from pyzbar import pyzbar

    # Decode QR code
    if isinstance(qr_image_data, str):
        img = Image.open(qr_image_data)
    else:
        img = qr_image_data

    decoded_objects = pyzbar.decode(img)
    if not decoded_objects:
        return {"error": "No QR code found in image"}

    qr_json_string = decoded_objects[0].data.decode('utf-8')

    # Step 2: Parse the QR JSON
    qr_data = json.loads(qr_json_string)
    # Example: {"q": 1, "pts": 2.0, "s": "Yx8CBgc5DjAUVDdQCTcXNUcCA0hDalFFRQp0G0o="}

    # Step 3: Decode the encrypted metadata
    encrypted_metadata = qr_data.get('s')
    if not encrypted_metadata:
        return {"error": "QR code does not contain regeneration data"}

    metadata = QuestionQRCode.decrypt_question_data(encrypted_metadata)
    # Returns: {"question_type": "VirtualAddressParts", "seed": 12345, "version": "1.0"}

    # Step 4: Regenerate the answers using this module
    result = regenerate_from_metadata(
        question_type=metadata['question_type'],
        seed=metadata['seed'],
        version=metadata['version'],
        points=qr_data['pts']
    )

    # Step 5: Return formatted answer data for your grading UI
    return {
        "question_number": qr_data['q'],
        "points": qr_data['pts'],
        "question_type": result['question_type'],
        "seed": result['seed'],
        "version": result['version'],
        "answers": format_answers_for_ui(result['answer_objects'])
    }


def format_answers_for_ui(answer_objects):
    """
    Convert Answer objects to a format suitable for your web UI.

    Args:
        answer_objects: dict of Answer objects from question.answers

    Returns:
        list: Formatted answers for display
    """
    formatted = []

    for key, answer_obj in answer_objects.items():
        answer_dict = {
            "key": key,
            "value": answer_obj.value,
            "type": type(answer_obj).__name__
        }

        # Add tolerance if it's a numerical answer
        if hasattr(answer_obj, 'tolerance') and answer_obj.tolerance:
            answer_dict['tolerance'] = answer_obj.tolerance
            answer_dict['display'] = f"{answer_obj.value} ± {answer_obj.tolerance}"
        else:
            answer_dict['display'] = str(answer_obj.value)

        formatted.append(answer_dict)

    return formatted


# ============================================================================
# SIMPLER API: If your web UI already has the decoded metadata
# ============================================================================

def simple_regenerate(question_type: str, seed: int, version: str, points: float = 1.0):
    """
    Direct API: If your web UI already decoded the QR and extracted the metadata.

    This is the simplest integration - just call this function with the metadata!

    Args:
        question_type: Question class name from QR code
        seed: Random seed from QR code
        version: Version string from QR code
        points: Point value for the question

    Returns:
        dict: Answer data with formatted values

    Example:
        >>> # Your web UI extracts: type="Paging", seed=67890, version="1.0"
        >>> answers = simple_regenerate("Paging", 67890, "1.0", points=2.0)
        >>> print(answers)
        [
            {
                "key": "page_table_entry",
                "value": "0x3A2F",
                "display": "0x3A2F",
                "type": "StringAnswer"
            }
        ]
    """
    result = regenerate_from_metadata(question_type, seed, version, points)
    return format_answers_for_ui(result['answer_objects'])


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    # Set encryption key for testing
    os.environ['QUIZ_ENCRYPTION_KEY'] = '5vprrXbqp0E5zDGTRn1GEpw6jN7Yd74wnJp_lEv3zow='

    print("="*70)
    print("WEB UI INTEGRATION EXAMPLE")
    print("="*70)

    # Example 1: Full flow with QR image scanning
    print("\n1. Full flow: Scan QR image and regenerate answers")
    print("-" * 70)
    # (This would use an actual QR image in production)
    # result = handle_qr_scan("path/to/qr_image.png")
    # print(json.dumps(result, indent=2))

    # Example 2: Direct metadata to answers (recommended for web UI)
    print("\n2. Direct API: Metadata -> Answers")
    print("-" * 70)

    try:
        # Simulate: Your web UI decoded QR and got these values
        question_type = "VirtualAddressParts"
        seed = 12345
        version = "1.0"
        points = 2.0

        print(f"Input: type={question_type}, seed={seed}, version={version}, points={points}")

        # Call the simple API
        answers = simple_regenerate(question_type, seed, version, points)

        print(f"\nRegenerated {len(answers)} answer(s):")
        for ans in answers:
            print(f"  • {ans['key']}: {ans['display']}")
            if 'tolerance' in ans:
                print(f"    (tolerance: ±{ans['tolerance']})")

    except Exception as e:
        print(f"ERROR: {e}")

    print("\n" + "="*70)
    print("\nINTEGRATION SUMMARY:")
    print("  1. Install this repo alongside your web UI")
    print("  2. Import: from grade_from_qr import regenerate_from_metadata")
    print("  3. Set QUIZ_ENCRYPTION_KEY environment variable")
    print("  4. Decode QR -> Extract metadata -> Call regenerate_from_metadata()")
    print("  5. Display answers in your grading UI")
    print("="*70)
