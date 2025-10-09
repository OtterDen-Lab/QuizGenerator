"""
QR Code generation module for quiz questions.

This module generates QR codes containing question metadata (question number,
points value, etc.) that can be embedded in PDF output for scanning and
automated grading.
"""

import json
import tempfile
import logging
from io import BytesIO
from pathlib import Path

import segno

log = logging.getLogger(__name__)


class QuestionQRCode:
    """
    Generator for question metadata QR codes.

    QR codes encode question information in JSON format for easy parsing
    after scanning. They use high error correction (30% recovery) to ensure
    reliability when printed and scanned.
    """

    # QR code size in cm for LaTeX output (suitable for 200 DPI scanning)
    DEFAULT_SIZE_CM = 1.5

    # Error correction level: H = 30% recovery (highest level)
    ERROR_CORRECTION = 'H'

    @classmethod
    def generate_qr_data(cls, question_number: int, points_value: float, **extra_data) -> str:
        """
        Generate JSON string containing question metadata.

        Args:
            question_number: Sequential question number in the quiz
            points_value: Point value of the question
            **extra_data: Additional metadata to include (e.g., version, section)

        Returns:
            JSON string with question metadata

        Example:
            >>> QuestionQRCode.generate_qr_data(1, 5.0)
            '{"q": 1, "pts": 5.0}'

            >>> QuestionQRCode.generate_qr_data(2, 10, version="A", section="memory")
            '{"q": 2, "pts": 10, "version": "A", "section": "memory"}'
        """
        data = {
            "q": question_number,
            "pts": points_value
        }
        # Add any extra metadata
        data.update(extra_data)
        return json.dumps(data, separators=(',', ':'))

    @classmethod
    def generate_png_path(cls, question_number: int, points_value: float,
                         scale: int = 4, **extra_data) -> str:
        """
        Generate QR code and save as PNG file, returning the file path.

        This is used for LaTeX inclusion via \\includegraphics.
        The file is saved to a temporary location that LaTeX can access.

        Args:
            question_number: Sequential question number
            points_value: Point value of the question
            scale: Scale factor for PNG generation (higher = larger file, better quality)
            **extra_data: Additional metadata

        Returns:
            Path to generated PNG file
        """
        qr_data = cls.generate_qr_data(question_number, points_value, **extra_data)

        # Generate QR code with high error correction
        qr = segno.make(qr_data, error=cls.ERROR_CORRECTION)

        # Create temporary file for the PNG
        # We use a predictable name based on question number so LaTeX can find it
        temp_dir = Path(tempfile.gettempdir()) / "quiz_qrcodes"
        temp_dir.mkdir(exist_ok=True)

        png_path = temp_dir / f"qr_q{question_number}.png"

        # Save as PNG with appropriate scale
        qr.save(str(png_path), scale=scale, border=1)

        log.debug(f"Generated QR code for question {question_number} at {png_path}")

        return str(png_path)

    @classmethod
    def generate_png_bytes(cls, question_number: int, points_value: float,
                          scale: int = 4, **extra_data) -> bytes:
        """
        Generate QR code and return as PNG bytes.

        Useful for in-memory operations or when you need the raw image data.

        Args:
            question_number: Sequential question number
            points_value: Point value of the question
            scale: Scale factor for PNG generation
            **extra_data: Additional metadata

        Returns:
            PNG image data as bytes
        """
        qr_data = cls.generate_qr_data(question_number, points_value, **extra_data)

        # Generate QR code with high error correction
        qr = segno.make(qr_data, error=cls.ERROR_CORRECTION)

        # Save to BytesIO
        buffer = BytesIO()
        qr.save(buffer, kind='png', scale=scale, border=1)
        buffer.seek(0)

        return buffer.read()

    @classmethod
    def cleanup_temp_files(cls):
        """
        Clean up temporary QR code files.

        Call this after PDF generation is complete to remove temporary files.
        """
        temp_dir = Path(tempfile.gettempdir()) / "quiz_qrcodes"
        if temp_dir.exists():
            for qr_file in temp_dir.glob("qr_q*.png"):
                try:
                    qr_file.unlink()
                    log.debug(f"Cleaned up QR code file: {qr_file}")
                except Exception as e:
                    log.warning(f"Failed to clean up {qr_file}: {e}")
