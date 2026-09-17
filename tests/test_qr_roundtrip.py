import base64

import pytest

from QuizGenerator.generation.qrcode_generator import QuestionQRCode


def _encode_v1(question_type: str, seed: int, version: str, key: bytes) -> str:
    data_str = f"{question_type}:{seed}:{version}"
    data_bytes = data_str.encode("utf-8")
    key_bytes = key[:16]
    obfuscated = bytes(b ^ key_bytes[i % len(key_bytes)] for i, b in enumerate(data_bytes))
    return base64.urlsafe_b64encode(obfuscated).decode("ascii")


def test_qr_roundtrip_v2():
    key = b"test-key-for-quiz"
    encrypted = QuestionQRCode.encrypt_question_data(
        "TestQuestion",
        12345,
        version="1.0",
        config={"a": 1},
        context={"b": 2},
        points_value=5.0,
        key=key,
    )
    decoded = QuestionQRCode.decrypt_question_data(encrypted, key=key)
    assert decoded["question_type"] == "TestQuestion"
    assert decoded["seed"] == 12345
    assert decoded["version"] == "1.0"
    assert decoded["config"] == {"a": 1}
    assert decoded["context"] == {"b": 2}
    assert decoded["points"] == 5.0


def test_qr_roundtrip_v1_fallback():
    key = b"test-key-for-quiz"
    encrypted = _encode_v1("TestQuestion", 12345, "1.0", key)
    decoded = QuestionQRCode.decrypt_question_data(encrypted, key=key)
    assert decoded["question_type"] == "TestQuestion"
    assert decoded["seed"] == 12345
    assert decoded["version"] == "1.0"


@pytest.fixture(autouse=True)
def _restore_generated_key():
    original = QuestionQRCode._generated_key
    QuestionQRCode._generated_key = None
    yield
    QuestionQRCode._generated_key = original


def test_encryption_key_loads_from_default_token_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("QUIZ_ENCRYPTION_KEY", raising=False)
    token_dir = tmp_path / ".tokens"
    token_dir.mkdir()
    token_file = token_dir / "quizgenerator.env"
    token_file.write_text("QUIZ_ENCRYPTION_KEY=key-from-token-file\n", encoding="utf-8")

    assert QuestionQRCode.get_encryption_key() == b"key-from-token-file"


def test_environment_key_overrides_default_token_file(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("QUIZ_ENCRYPTION_KEY", "explicit-environment-key")
    token_dir = tmp_path / ".tokens"
    token_dir.mkdir()
    token_file = token_dir / "quizgenerator.env"
    token_file.write_text("QUIZ_ENCRYPTION_KEY=key-from-token-file\n", encoding="utf-8")

    assert QuestionQRCode.get_encryption_key() == b"explicit-environment-key"
