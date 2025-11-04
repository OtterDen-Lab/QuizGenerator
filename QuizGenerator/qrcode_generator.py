"""
QR Code generation module for quiz questions.

This module generates QR codes containing question metadata (question number,
points value, etc.) that can be embedded in PDF output for scanning and
automated grading.

The QR codes include encrypted data that allows regenerating question answers
without storing separate files, enabling efficient grading of randomized exams.
"""

import json
import tempfile
import logging
import os
import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, Any

import segno
from cryptography.fernet import Fernet

log = logging.getLogger(__name__)


class QuestionQRCode:
    """
    Generator for question metadata QR codes.

    QR codes encode question information in JSON format for easy parsing
    after scanning. They use high error correction (30% recovery) to ensure
    reliability when printed and scanned.
    """

    # QR code size in cm for LaTeX output (suitable for 200 DPI scanning)
    DEFAULT_SIZE_CM = 1.5  # Compact size suitable for ~60 char encoded data

    # Error correction level: M = 15% recovery (balanced for compact encoded data)
    ERROR_CORRECTION = 'M'

    @classmethod
    def get_encryption_key(cls) -> bytes:
        """
        Get encryption key from environment or generate new one.

        The key is loaded from QUIZ_ENCRYPTION_KEY environment variable.
        If not set, generates a new key (for development only).

        Returns:
            bytes: Fernet encryption key

        Note:
            In production, always set QUIZ_ENCRYPTION_KEY environment variable!
            Generate a key once with: Fernet.generate_key()
        """
        key_str = os.environ.get('QUIZ_ENCRYPTION_KEY')

        if key_str is None:
            log.warning(
                "QUIZ_ENCRYPTION_KEY not set! Generating temporary key. "
                "Set this environment variable for production use!"
            )
            # Generate temporary key for development
            return Fernet.generate_key()

        # Key should be stored as base64 string in env
        return key_str.encode()

    @classmethod
    def encrypt_question_data(cls, question_type: str, seed: int, version: str,
                              config: Optional[Dict[str, Any]] = None,
                              key: Optional[bytes] = None) -> str:
        """
        Encode question regeneration data with optional simple obfuscation.

        Args:
            question_type: Class name of the question (e.g., "VectorDotProduct")
            seed: Random seed used to generate this specific question
            version: Question class version (e.g., "1.0")
            config: Optional dictionary of configuration parameters
            key: Encryption key (uses environment key if None)

        Returns:
            str: Base64-encoded (optionally XOR-obfuscated) data

        Example:
            >>> encrypted = QuestionQRCode.encrypt_question_data("VectorDot", 12345, "1.0")
            >>> print(encrypted)
            'VmVjdG9yRG90OjEyMzQ1OjEuMA=='
        """
        # Create compact data string, including config if provided
        if config:
            # Serialize config as JSON and append to data string
            config_json = json.dumps(config, separators=(',', ':'))
            data_str = f"{question_type}:{seed}:{version}:{config_json}"
        else:
            data_str = f"{question_type}:{seed}:{version}"
        data_bytes = data_str.encode('utf-8')

        # Simple XOR obfuscation if key is provided (optional, for basic protection)
        if key is None:
            key = cls.get_encryption_key()

        if key:
            # Use first 16 bytes of key for simple XOR obfuscation
            key_bytes = key[:16] if isinstance(key, bytes) else key.encode()[:16]
            # XOR each byte with repeating key pattern
            obfuscated = bytes(b ^ key_bytes[i % len(key_bytes)] for i, b in enumerate(data_bytes))
        else:
            obfuscated = data_bytes

        # Base64 encode for compact representation
        encoded = base64.urlsafe_b64encode(obfuscated).decode('utf-8')

        log.debug(f"Encoded question data: {question_type} seed={seed} version={version} ({len(encoded)} chars)")

        return encoded

    @classmethod
    def decrypt_question_data(cls, encrypted_data: str,
                             key: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Decode question regeneration data from QR code.

        Args:
            encrypted_data: Base64-encoded (optionally XOR-obfuscated) string from QR code
            key: Encryption key (uses environment key if None)

        Returns:
            dict: {"question_type": str, "seed": int, "version": str, "config": dict (optional)}

        Raises:
            ValueError: If decoding fails or data is malformed

        Example:
            >>> data = QuestionQRCode.decrypt_question_data("VmVjdG9yRG90OjEyMzQ1OjEuMA==")
            >>> print(data)
            {"question_type": "VectorDot", "seed": 12345, "version": "1.0"}
        """
        if key is None:
            key = cls.get_encryption_key()

        try:
            # Decode from base64
            obfuscated = base64.urlsafe_b64decode(encrypted_data.encode())

            # Reverse XOR obfuscation if key is provided
            if key:
                key_bytes = key[:16] if isinstance(key, bytes) else key.encode()[:16]
                data_bytes = bytes(b ^ key_bytes[i % len(key_bytes)] for i, b in enumerate(obfuscated))
            else:
                data_bytes = obfuscated

            data_str = data_bytes.decode('utf-8')

            # Parse data string - can be 3 or 4 parts (4th is optional config)
            parts = data_str.split(':', 3)  # Split into max 4 parts
            if len(parts) < 3:
                raise ValueError(f"Invalid encoded data format: expected at least 3 parts, got {len(parts)}")

            question_type = parts[0]
            seed_str = parts[1]
            version = parts[2]

            result = {
                "question_type": question_type,
                "seed": int(seed_str),
                "version": version
            }

            # Parse config JSON if present
            if len(parts) == 4:
                try:
                    result["config"] = json.loads(parts[3])
                except json.JSONDecodeError as e:
                    log.warning(f"Failed to parse config JSON: {e}")
                    # Continue without config rather than failing

            return result

        except Exception as e:
            log.error(f"Failed to decode question data: {e}")
            raise ValueError(f"Failed to decode QR code data: {e}")

    @classmethod
    def generate_qr_data(cls, question_number: int, points_value: float, **extra_data) -> str:
        """
        Generate JSON string containing question metadata.

        Args:
            question_number: Sequential question number in the quiz
            points_value: Point value of the question
            **extra_data: Additional metadata to include
                - question_type (str): Question class name for regeneration
                - seed (int): Random seed used for this question
                - version (str): Question class version
                - config (dict): Question-specific configuration parameters

        Returns:
            JSON string with question metadata

        Example:
            >>> QuestionQRCode.generate_qr_data(1, 5.0)
            '{"q": 1, "pts": 5.0}'

            >>> QuestionQRCode.generate_qr_data(
            ...     2, 10,
            ...     question_type="VectorDot",
            ...     seed=12345,
            ...     version="1.0",
            ...     config={"max_value": 100}
            ... )
            '{"q": 2, "pts": 10, "s": "gAAAAAB..."}'
        """
        data = {
            "q": question_number,
            "pts": points_value
        }

        # If question regeneration data provided, encrypt it
        if all(k in extra_data for k in ['question_type', 'seed', 'version']):
            # Include config in encrypted data if present
            config = extra_data.get('config', {})
            encrypted = cls.encrypt_question_data(
                extra_data['question_type'],
                extra_data['seed'],
                extra_data['version'],
                config=config
            )
            data['s'] = encrypted

            # Remove the unencrypted data
            extra_data = {k: v for k, v in extra_data.items()
                         if k not in ['question_type', 'seed', 'version', 'config']}

        # Add any remaining extra metadata
        data.update(extra_data)

        return json.dumps(data, separators=(',', ':'))

    @classmethod
    def generate_qr_pdf(cls, question_number: int, points_value: float,
                         scale: int = 10, **extra_data) -> str:
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

        qr_path = temp_dir / f"qr_q{question_number}.pdf"

        # Save as PNG with appropriate scale
        qr.save(str(qr_path), scale=scale, border=0)

        log.debug(f"Generated QR code for question {question_number} at {qr_path}")

        return str(qr_path)

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
