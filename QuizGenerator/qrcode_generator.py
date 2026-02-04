"""
QR Code generation module for quiz questions.

This module generates QR codes containing question metadata (question number,
points value, etc.) that can be embedded in PDF output for scanning and
automated grading.

The QR codes include encrypted data that allows regenerating question answers
without storing separate files, enabling efficient grading of randomized exams.
"""

import base64
import hashlib
import json
import logging
import os
import tempfile
import zlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import segno
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

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
    _generated_key: Optional[bytes] = None
    V2_PREFIX = "v2."

    @classmethod
    def _persist_generated_key(cls, key: bytes) -> None:
        try:
            os.makedirs("out/keys", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            path = os.path.join("out", "keys", f"quiz_encryption_key-{timestamp}.log")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("QUIZ_ENCRYPTION_KEY=")
                handle.write(key.decode("ascii"))
                handle.write("\n")
            log.warning(f"Wrote generated QUIZ_ENCRYPTION_KEY to {path}")
        except Exception as exc:
            log.warning(f"Failed to persist generated QUIZ_ENCRYPTION_KEY: {exc}")

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
            if cls._generated_key is None:
                cls._generated_key = Fernet.generate_key()
                os.environ["QUIZ_ENCRYPTION_KEY"] = cls._generated_key.decode("ascii")
                cls._persist_generated_key(cls._generated_key)
            return cls._generated_key

        # Key should be stored as base64 string in env
        return key_str.encode()

    @classmethod
    def _derive_aead_key(cls, key: bytes) -> bytes:
        return hashlib.sha256(key).digest()

    @classmethod
    def _encrypt_v2(cls, payload: Dict[str, Any], *, key: Optional[bytes] = None) -> str:
        if key is None:
            key = cls.get_encryption_key()
        aead_key = cls._derive_aead_key(key)
        aesgcm = AESGCM(aead_key)
        nonce = os.urandom(12)
        json_bytes = json.dumps(payload, separators=(',', ':'), ensure_ascii=False).encode("utf-8")
        compressed = zlib.compress(json_bytes)
        ciphertext = aesgcm.encrypt(nonce, compressed, None)
        token = base64.urlsafe_b64encode(nonce + ciphertext).decode("ascii")
        return f"{cls.V2_PREFIX}{token}"

    @classmethod
    def _decrypt_v2(cls, encrypted_data: str, *, key: Optional[bytes] = None) -> Dict[str, Any]:
        if not encrypted_data.startswith(cls.V2_PREFIX):
            raise ValueError("Not a v2 payload")
        if key is None:
            key = cls.get_encryption_key()
        aead_key = cls._derive_aead_key(key)
        token = encrypted_data[len(cls.V2_PREFIX):]
        raw = base64.urlsafe_b64decode(token.encode("ascii"))
        nonce, ciphertext = raw[:12], raw[12:]
        aesgcm = AESGCM(aead_key)
        compressed = aesgcm.decrypt(nonce, ciphertext, None)
        json_bytes = zlib.decompress(compressed)
        return json.loads(json_bytes.decode("utf-8"))

    @classmethod
    def encrypt_question_data(cls, question_type: str, seed: int, version: Optional[str] = None,
                              config: Optional[Dict[str, Any]] = None,
                              context: Optional[Dict[str, Any]] = None,
                              points_value: Optional[float] = None,
                              key: Optional[bytes] = None) -> str:
        """
        Encode question regeneration data for QR embedding.

        Args:
            question_type: Class name of the question (e.g., "VectorDotProduct")
            seed: Random seed used to generate this specific question
            version: Optional question version string
            config: Optional dictionary of configuration parameters
            context: Optional dictionary of context extras
            points_value: Optional points value (for redundancy)
            key: Encryption key (uses environment key if None)

        Returns:
            str: Base64-encoded encrypted payload with v2 prefix

        Example:
            >>> encrypted = QuestionQRCode.encrypt_question_data("VectorDot", 12345, "1.0")
            >>> print(encrypted)
            'VmVjdG9yRG90OjEyMzQ1OjEuMA=='
        """
        payload: Dict[str, Any] = {
            "t": question_type,
            "s": seed,
        }
        if points_value is not None:
            payload["p"] = points_value
        if config:
            payload["c"] = config
        if context:
            payload["x"] = context
        if version:
            payload["v"] = version

        encoded = cls._encrypt_v2(payload, key=key)
        log.debug(f"Encoded question data v2: {question_type} seed={seed} ({len(encoded)} chars)")
        return encoded

    @classmethod
    def decrypt_question_data(cls, encrypted_data: str,
                             key: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Decode question regeneration data from QR code (v2 preferred, v1 fallback).

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
            if encrypted_data.startswith(cls.V2_PREFIX):
                payload = cls._decrypt_v2(encrypted_data, key=key)
                result = {
                    "question_type": payload.get("t"),
                    "seed": int(payload.get("s")),
                }
                if "v" in payload:
                    result["version"] = payload.get("v")
                if "c" in payload:
                    result["config"] = payload.get("c")
                if "x" in payload:
                    result["context"] = payload.get("x")
                if "p" in payload:
                    result["points"] = payload.get("p")
                return result

            # V1 fallback (XOR obfuscation)
            obfuscated = base64.urlsafe_b64decode(encrypted_data.encode())
            if key:
                key_bytes = key[:16] if isinstance(key, bytes) else key.encode()[:16]
                data_bytes = bytes(b ^ key_bytes[i % len(key_bytes)] for i, b in enumerate(obfuscated))
            else:
                data_bytes = obfuscated

            data_str = data_bytes.decode('utf-8')
            parts = data_str.split(':', 3)
            if len(parts) < 2:
                raise ValueError(f"Invalid encoded data format: expected at least 2 parts, got {len(parts)}")

            question_type = parts[0]
            seed_str = parts[1]
            version = parts[2] if len(parts) >= 3 else None

            result = {
                "question_type": question_type,
                "seed": int(seed_str),
            }
            if version:
                result["version"] = version

            if len(parts) == 4:
                try:
                    result["config"] = json.loads(parts[3])
                except json.JSONDecodeError as e:
                    log.warning(f"Failed to parse config JSON: {e}")

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
            "p": points_value
        }

        # If question regeneration data provided, encrypt it
        if all(k in extra_data for k in ['question_type', 'seed']):
            config = extra_data.get('config', {})
            context = extra_data.get('context', {})
            encrypted = cls.encrypt_question_data(
                extra_data['question_type'],
                extra_data['seed'],
                extra_data.get('version'),
                config=config,
                context=context,
                points_value=points_value
            )
            data['s'] = encrypted

            extra_data = {
                k: v for k, v in extra_data.items()
                if k not in ['question_type', 'seed', 'version', 'config', 'context']
            }

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
