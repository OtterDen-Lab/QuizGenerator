#!/usr/bin/env python
"""
QR Code-based Grading Utility

This script scans QR codes from quiz PDFs to regenerate question answers
for grading. It supports both scanning from image files and interactive
scanning from webcam/scanner.

CLI Usage:
    # Scan a single QR code image
    quizregen --image qr_code.png

    # Scan QR codes from a scanned exam page
    quizregen --image exam_page.jpg --all

    # Decode an encrypted string directly
    python -m QuizGenerator.regenerate --encrypted-str "EzE6JF86CDlf..."

    # Decode with custom point value
    python -m QuizGenerator.regenerate --encrypted-str "EzE6JF86CDlf..." --points 5.0

API Usage (recommended for web UIs):
    from QuizGenerator.regenerate import regenerate_from_encrypted

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

import base64
import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Literal

import typer
import yaml

# Load environment variables from .env file
try:
  from dotenv import load_dotenv
  # Try loading from current directory first, then home directory
  if os.path.exists('.env'):
    load_dotenv('.env')
  else:
    load_dotenv(os.path.join(os.path.expanduser("~"), ".env"))
except ImportError:
  # dotenv not available, will use system environment variables only
  pass

# Quiz generator imports (always available)
from QuizGenerator.qrcode_generator import QuestionQRCode
from QuizGenerator.question import QuestionGroup, QuestionRegistry
from QuizGenerator.quiz import Quiz

# QR code reading (optional - only needed for CLI usage with --image)
# Your web UI should use its own QR decoding library
try:
  from PIL import Image
  from pyzbar import pyzbar
  
  HAS_IMAGE_SUPPORT = True
except ImportError:
  HAS_IMAGE_SUPPORT = False
  # Don't fail immediately - only fail if user tries to use --image flag

logging.basicConfig(
  level=logging.INFO,
  format='%(levelname)s: %(message)s'
)
log = logging.getLogger(__name__)


def scan_qr_from_image(image_path: str) -> list[str]:
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


def parse_qr_data(qr_string: str) -> dict[str, Any]:
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


def _inline_image_upload(img_data) -> str:
  img_data.seek(0)
  b64 = base64.b64encode(img_data.read()).decode("ascii")
  return f"data:image/png;base64,{b64}"


def _resolve_upload_func(
  image_mode: str,
  upload_func: Callable | None
) -> Callable | None:
  if image_mode == "inline":
    return _inline_image_upload
  if image_mode == "upload":
    if upload_func is None:
      raise ValueError("image_mode='upload' requires upload_func")
    return upload_func
  if image_mode == "none":
    return None
  raise ValueError(f"Unknown image_mode: {image_mode}")


def _render_html(element, upload_func=None, **kwargs) -> str:
  if upload_func is None:
    return element.render("html", **kwargs)
  return element.render("html", upload_func=upload_func, **kwargs)


def _load_yaml_docs(
  *,
  yaml_path: str | None = None,
  yaml_text: str | None = None,
  yaml_docs: list[dict] | None = None
) -> list[dict]:
  if yaml_docs is not None:
    return yaml_docs
  if yaml_path is not None:
    with open(yaml_path, "r", encoding="utf-8") as handle:
      return list(yaml.safe_load_all(handle))
  if yaml_text is not None:
    return list(yaml.safe_load_all(yaml_text))
  raise ValueError("Must provide yaml_path, yaml_text, or yaml_docs.")


def _find_questions_by_id(quizzes: list[Quiz], question_id: str) -> list:
  matches = []
  for quiz in quizzes:
    for item in quiz.questions:
      if isinstance(item, QuestionGroup):
        for question in item.questions:
          if getattr(question, "question_id", None) == question_id:
            matches.append(question)
      else:
        if getattr(item, "question_id", None) == question_id:
          matches.append(item)
  return matches


def regenerate_from_yaml_metadata(
  *,
  question_id: str,
  seed: int,
  points: float = 1.0,
  yaml_id: str | None = None,
  yaml_path: str | None = None,
  yaml_text: str | None = None,
  yaml_docs: list[dict] | None = None,
  image_mode: str = "inline",
  upload_func: Callable | None = None
) -> dict[str, Any]:
  """
  Regenerate question answers using YAML + question_id + seed.

  Args:
      question_id: Question identifier from QR payload
      seed: Random seed used to generate the question
      points: Point value for the question (default: 1.0)
      yaml_id: Optional YAML identifier from QR payload (used for warnings)
      yaml_path/yaml_text/yaml_docs: YAML source defining the questions
      image_mode: "inline", "upload", or "none" for HTML image handling
      upload_func: Optional upload function used when image_mode="upload"
  """
  docs = _load_yaml_docs(yaml_path=yaml_path, yaml_text=yaml_text, yaml_docs=yaml_docs)
  quizzes = Quiz.from_exam_dicts(copy.deepcopy(docs), source_path=yaml_path)

  matches = _find_questions_by_id(quizzes, question_id)
  if not matches:
    raise ValueError(f"Question ID '{question_id}' not found in provided YAML.")
  if len(matches) > 1:
    raise ValueError(f"Question ID '{question_id}' is not unique in provided YAML.")

  question = matches[0]

  warnings: list[str] = []
  yaml_ids = {
    doc.get("yaml_id") for doc in docs
    if isinstance(doc, dict) and doc.get("yaml_id")
  }
  if yaml_id and yaml_ids and yaml_id not in yaml_ids:
    warnings.append(f"YAML id mismatch: QR has {yaml_id}, YAML has {sorted(yaml_ids)}")

  if points is not None and getattr(question, "points_value", None) != points:
    warnings.append(
      f"Points mismatch: QR has {points}, YAML has {getattr(question, 'points_value', None)}"
    )

  instance = question.instantiate(rng_seed=seed)
  question_ast = question._build_question_ast(instance)

  answer_kind, canvas_answers = question._answers_for_canvas(
    instance.answers,
    instance.can_be_numerical
  )

  resolved_upload_func = _resolve_upload_func(image_mode, upload_func)

  question_html = _render_html(
    question_ast.body,
    show_answers=True,
    upload_func=resolved_upload_func
  )

  explanation_markdown = question_ast.explanation.render("markdown")
  if not explanation_markdown or "[Please reach out to your professor for clarification]" in explanation_markdown:
    explanation_markdown = None

  explanation_html = _render_html(
    question_ast.explanation,
    upload_func=resolved_upload_func
  )
  if not explanation_html or "[Please reach out to your professor for clarification]" in explanation_html:
    explanation_html = None

  result = {
    "question_id": question_id,
    "yaml_id": yaml_id,
    "question_type": question._get_registered_name(),
    "version": getattr(question, "VERSION", None),
    "seed": seed,
    "points": points,
    "answers": {
      "kind": answer_kind.value,
      "data": canvas_answers
    },
    "answer_objects": instance.answers,
    "answer_key_html": question_html,
    "explanation_markdown": explanation_markdown,
    "explanation_html": explanation_html
  }
  if warnings:
    result["warnings"] = warnings
  return result


def regenerate_question_answer(
  qr_data: dict[str, Any],
  *,
  image_mode: str = "inline",
  upload_func: Callable | None = None,
  yaml_path: str | None = None,
  yaml_text: str | None = None,
  yaml_docs: list[dict] | None = None
) -> dict[str, Any]:
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
          "answers": dict,
      "explanation_markdown": str | None  # Markdown explanation (None if not available)
      "explanation_html": str | None  # HTML explanation (None if not available)
  }
  """
  question_num = qr_data.get('q')
  points = qr_data.get('p')
  if points is None:
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

    question_id = regen_data.get('question_id')
    seed = regen_data.get('seed')

    if question_id:
      yaml_id = regen_data.get('yaml_id')
      result['question_id'] = question_id
      result['yaml_id'] = yaml_id
      result['seed'] = seed

      if not (yaml_path or yaml_text or yaml_docs):
        log.warning(f"Question {question_num}: YAML is required to regenerate question_id '{question_id}'.")
        return result

      regen_result = regenerate_from_yaml_metadata(
        question_id=question_id,
        seed=seed,
        points=points,
        yaml_id=yaml_id,
        yaml_path=yaml_path,
        yaml_text=yaml_text,
        yaml_docs=yaml_docs,
        image_mode=image_mode,
        upload_func=upload_func
      )
      regen_result["question_number"] = question_num
      return regen_result

    question_type = regen_data['question_type']
    version = regen_data.get('version')
    config = regen_data.get('config', {})
    context_extras = regen_data.get('context', {})

    result['question_type'] = question_type
    result['seed'] = seed
    result['version'] = version
    if config:
      result['config'] = config

    if version:
      log.info(f"Question {question_num}: {question_type} (seed={seed}, version={version})")
    else:
      log.info(f"Question {question_num}: {question_type} (seed={seed})")
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
    instance = question.instantiate(rng_seed=seed, **context_extras)
    question_ast = question._build_question_ast(instance)

    # Extract answers
    answer_kind, canvas_answers = question._answers_for_canvas(
      instance.answers,
      instance.can_be_numerical
    )

    result['answers'] = {
      'kind': answer_kind.value,
      'data': canvas_answers
    }

    # Also store the raw answer objects for easier access
    result['answer_objects'] = instance.answers

    resolved_upload_func = _resolve_upload_func(image_mode, upload_func)

    # Generate HTML answer key for grading
    question_html = _render_html(
      question_ast.body,
      show_answers=True,
      upload_func=resolved_upload_func
    )
    result['answer_key_html'] = question_html

    # Generate markdown explanation for students
    explanation_markdown = question_ast.explanation.render("markdown")
    # Return None if explanation is empty or contains the default placeholder
    if not explanation_markdown or "[Please reach out to your professor for clarification]" in explanation_markdown:
      result['explanation_markdown'] = None
    else:
      result['explanation_markdown'] = explanation_markdown

    # Generate HTML explanation (optional for web UIs)
    explanation_html = _render_html(
      question_ast.explanation,
      upload_func=resolved_upload_func
    )
    if not explanation_html or "[Please reach out to your professor for clarification]" in explanation_html:
      result["explanation_html"] = None
    else:
      result["explanation_html"] = explanation_html

    log.info(f"  Successfully regenerated question with {len(canvas_answers)} answer(s)")

    return result
  
  except Exception as e:
    log.error(f"Failed to regenerate question {question_num}: {e}")
    import traceback
    log.debug(traceback.format_exc())
    return result


def regenerate_from_encrypted(
  encrypted_data: str,
  points: float = 1.0,
  *,
  image_mode: str = "inline",
  upload_func: Callable | None = None,
  yaml_path: str | None = None,
  yaml_text: str | None = None,
  yaml_docs: list[dict] | None = None
) -> dict[str, Any]:
  """
  Regenerate question answers from encrypted QR code data (RECOMMENDED API).

  This is the simplest function for web UI integration - just pass the encrypted
  string from the QR code and get back the complete answer key.

  Args:
      encrypted_data: The encrypted 's' field from the QR code JSON
      points: Point value for the question (default: 1.0)
      image_mode: "inline", "upload", or "none" for HTML image handling
      upload_func: Optional upload function used when image_mode="upload"
      yaml_path/yaml_text/yaml_docs: Required when QR payload uses question_id (YAML-based regeneration)

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
          "answer_key_html": str,  # HTML rendering of question with answers shown
          "explanation_markdown": str | None  # Markdown explanation (None if not available)
          "explanation_html": str | None  # HTML explanation (None if not available)
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
  
  question_id = decrypted.get("question_id")
  seed = decrypted.get("seed")

  if question_id:
    return regenerate_from_yaml_metadata(
      question_id=question_id,
      seed=seed,
      points=points,
      yaml_id=decrypted.get("yaml_id"),
      yaml_path=yaml_path,
      yaml_text=yaml_text,
      yaml_docs=yaml_docs,
      image_mode=image_mode,
      upload_func=upload_func
    )

  # Extract fields for legacy regeneration
  question_type = decrypted['question_type']
  version = decrypted['version']
  kwargs = decrypted.get('config', {})

  # Use the existing regeneration logic
  return regenerate_from_metadata(
    question_type,
    seed,
    version,
    points,
    kwargs,
    image_mode=image_mode,
    upload_func=upload_func
  )


def regenerate_from_metadata(
    question_type: str, seed: int, version: str,
    points: float = 1.0, kwargs: dict[str, Any] | None = None,
    *, image_mode: str = "inline", upload_func: Callable | None = None
) -> dict[str, Any]:
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
      image_mode: "inline", "upload", or "none" for HTML image handling
      upload_func: Optional upload function used when image_mode="upload"

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
    instance = question.instantiate(rng_seed=seed)
    question_ast = question._build_question_ast(instance)
    
    # Extract answers
    answer_kind, canvas_answers = question._answers_for_canvas(
      instance.answers,
      instance.can_be_numerical
    )
    
    resolved_upload_func = _resolve_upload_func(image_mode, upload_func)

    # Generate HTML answer key for grading
    question_html = _render_html(
      question_ast.body,
      show_answers=True,
      upload_func=resolved_upload_func
    )

    # Generate markdown explanation for students
    explanation_markdown = question_ast.explanation.render("markdown")
    # Return None if explanation is empty or contains the default placeholder
    if not explanation_markdown or "[Please reach out to your professor for clarification]" in explanation_markdown:
      explanation_markdown = None

    explanation_html = _render_html(
      question_ast.explanation,
      upload_func=resolved_upload_func
    )
    if not explanation_html or "[Please reach out to your professor for clarification]" in explanation_html:
      explanation_html = None

    result = {
      "question_type": question_type,
      "seed": seed,
      "version": version,
      "points": points,
      "answers": {
        "kind": answer_kind.value,
        "data": canvas_answers
      },
      "answer_objects": instance.answers,
      "answer_key_html": question_html,
      "explanation_markdown": explanation_markdown,
      "explanation_html": explanation_html
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


def display_answer_summary(question_data: dict[str, Any]) -> None:
  """
  Display a formatted summary of the question and its answer(s).

  Args:
      question_data: Question data dictionary from regenerate_question_answer
  """
  print("\n" + "=" * 60)
  print(f"Question {question_data['question_number']}: {question_data.get('points', '?')} points")
  
  if 'question_type' in question_data:
    print(f"Type: {question_data['question_type']}")
    print(f"Seed: {question_data['seed']}")
    if question_data.get('version') is not None:
      print(f"Version: {question_data['version']}")
  
  if 'answer_objects' in question_data:
    print("\nANSWERS:")
    answer_objects = question_data['answer_objects']
    if isinstance(answer_objects, dict):
      for key, answer_obj in answer_objects.items():
        print(f"  {key}: {answer_obj.value}")
        if hasattr(answer_obj, 'tolerance') and answer_obj.tolerance:
          print(f"    (tolerance: ±{answer_obj.tolerance})")
    else:
      for i, answer_obj in enumerate(answer_objects, start=1):
        print(f"  {i}: {answer_obj.value}")
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

  if 'explanation_markdown' in question_data and question_data['explanation_markdown'] is not None:
    print("Markdown explanation available in result['explanation_markdown']")

  if 'explanation_html' in question_data and question_data['explanation_html'] is not None:
    print("HTML explanation available in result['explanation_html']")

  print("=" * 60)


app = typer.Typer(
  add_completion=True,
  no_args_is_help=False,
  invoke_without_command=True,
  help="Scan quiz QR codes and regenerate answers for grading."
)


@app.callback()
def _cli(
  ctx: typer.Context,
  image: str | None = typer.Option(None, "--image", help="Path to image file containing QR code(s)."),
  encrypted_str: str | None = typer.Option(
    None,
    "--encrypted-str",
    help="Encrypted string from QR code to decode directly.",
  ),
  yaml_path: str | None = typer.Option(
    None,
    "--yaml",
    help="Path to replay YAML (required for question_id-based regeneration).",
  ),
  points: float = typer.Option(
    1.0,
    "--points",
    help="Point value for the question (used with --encrypted-str).",
  ),
  process_all: bool = typer.Option(
    False,
    "--all",
    help="Process all QR codes found in the image (default: only first one).",
  ),
  output: str | None = typer.Option(None, "--output", help="Save results to JSON file."),
  verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose debug logging."),
  image_mode: Literal["inline", "none"] = typer.Option(
    "inline",
    "--image-mode",
    help="HTML image handling mode.",
  ),
) -> None:
  if ctx.resilient_parsing:
    return

  if verbose:
    logging.getLogger().setLevel(logging.DEBUG)

  if not image and not encrypted_str:
    typer.echo(ctx.get_help())
    typer.secho("\nERROR: Either --image or --encrypted-str is required", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)

  if image and encrypted_str:
    typer.secho(
      "ERROR: Cannot use both --image and --encrypted-str at the same time",
      fg=typer.colors.RED,
      err=True,
    )
    raise typer.Exit(code=1)

  results = []

  if encrypted_str:
    try:
      log.info(f"Decoding encrypted string (points={points})")
      result = regenerate_from_encrypted(
        encrypted_str,
        points,
        image_mode=image_mode,
        yaml_path=yaml_path
      )

      if "answers" not in result:
        typer.secho(
          "ERROR: This QR payload requires a replay YAML. Pass --yaml <replay.yaml>.",
          fg=typer.colors.RED,
          err=True,
        )
        raise typer.Exit(code=1)

      question_data = {
        "question_number": "N/A",
        "points": points,
        "question_type": result["question_type"],
        "seed": result["seed"],
        "version": result.get("version"),
        "answers": result["answers"],
        "answer_objects": result["answer_objects"],
        "answer_key_html": result["answer_key_html"],
        "explanation_markdown": result.get("explanation_markdown"),
        "explanation_html": result.get("explanation_html")
      }

      if "kwargs" in result:
        question_data["config"] = result["kwargs"]

      results.append(question_data)
      display_answer_summary(question_data)

    except typer.Exit:
      raise
    except Exception as e:
      log.error(f"Failed to decode encrypted string: {e}")
      import traceback
      if verbose:
        log.debug(traceback.format_exc())
      raise typer.Exit(code=1) from e

  else:
    qr_codes = scan_qr_from_image(image)

    if not qr_codes:
      typer.secho("No QR codes found in image", fg=typer.colors.RED, err=True)
      raise typer.Exit(code=1)

    if not process_all:
      qr_codes = qr_codes[:1]
      log.info("Processing only the first QR code (use --all to process all)")

    for qr_string in qr_codes:
      qr_data = parse_qr_data(qr_string)
      if not qr_data:
        continue

      question_data = regenerate_question_answer(
        qr_data,
        image_mode=image_mode,
        yaml_path=yaml_path
      )

      if question_data:
        if "answers" not in question_data:
          typer.secho(
            "ERROR: This QR payload requires a replay YAML. Pass --yaml <replay.yaml>.",
            fg=typer.colors.RED,
            err=True,
          )
          raise typer.Exit(code=1)
        results.append(question_data)
        display_answer_summary(question_data)

  if output:
    output_path = Path(output)
    with open(output_path, "w", encoding="utf-8") as f:
      json.dump(results, f, indent=2, default=str)
    log.info(f"Results saved to {output_path}")

  if not results:
    typer.secho("\nNo questions could be regenerated from the QR codes.", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)

  typer.echo(f"\nSuccessfully regenerated {len(results)} question(s)")


def main(argv: list[str] | None = None) -> None:
  if argv is None:
    app()
    return

  original_argv = sys.argv
  try:
    sys.argv = [original_argv[0] if original_argv else "quizregen", *argv]
    app()
  finally:
    sys.argv = original_argv


if __name__ == '__main__':
  main()
