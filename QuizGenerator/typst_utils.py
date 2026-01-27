#!/usr/bin/env python3
"""
Typst Utilities for Question Measurement

Provides utilities to measure question heights using Typst's layout engine,
enabling accurate bin-packing for PDF generation.
"""

import json
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Optional
import QuizGenerator.contentast as ca

import logging
log = logging.getLogger(__name__)


def measure_typst_content(typst_content: str, page_width_cm: float = 18.0) -> Optional[float]:
    """
    Measure the height of Typst content by compiling and querying.

    Args:
        typst_content: Typst markup to measure
        page_width_cm: Page width for measurement context (default 18cm)

    Returns:
        Height in cm, or None if measurement fails
    """

    # Get the Typst header which includes fillline and other helper functions
    typst_header = ca.Document.TYPST_HEADER

    # Create temporary Typst file with measurement wrapper
    typst_code = textwrap.dedent(f"""
      {typst_header}
      
      #set page(width: {page_width_cm}cm, height: auto, margin: 0cm)
      
      #let content_to_measure = [{typst_content}]
      
      #context {{
        let measured = measure(content_to_measure)
      
        [#metadata((
          height_pt: measured.height.pt(),
          height_cm: measured.height.to-absolute().cm(),
          width_pt: measured.width.pt(),
          width_cm: measured.width.to-absolute().cm(),
        )) <measurement>]
      }}
      
      // Render the content (required for compilation)
      #content_to_measure
      """
    )

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.typ', delete=False) as f:
        f.write(typst_code)
        temp_file = Path(f.name)

    try:
        # Query for measurements
        result = subprocess.run(
            ['typst', 'query', str(temp_file), '<measurement>', '--format', 'json'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            log.warning(f"Typst query failed: {result.stderr}")
            return None

        # Parse JSON output
        measurements = json.loads(result.stdout)

        if measurements:
            return measurements[0]['value']['height_cm']
        else:
            log.warning("No measurements found in Typst output")
            return None

    except subprocess.TimeoutExpired:
        log.warning("Typst measurement timed out")
        return None
    except Exception as e:
        log.warning(f"Failed to measure Typst content: {e}")
        return None
    finally:
        # Clean up temp file
        temp_file.unlink(missing_ok=True)


def check_typst_available() -> bool:
    """
    Check if Typst is available on the system.

    Returns:
        True if Typst is available, False otherwise
    """
    try:
        result = subprocess.run(
            ['typst', '--version'],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
