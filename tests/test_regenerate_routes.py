from unittest.mock import Mock, patch

from QuizGenerator.regenerate import (
    regenerate_from_encrypted,
    regenerate_question_answer,
)


def test_regenerate_from_encrypted_routes_to_yaml_metadata():
    decrypted = {
        "question_id": "q-seeded-1",
        "yaml_id": "yaml-alpha",
        "seed": 123,
    }
    expected = {"question_type": "fromtext", "seed": 123}

    with patch(
        "QuizGenerator.regenerate.QuestionQRCode.decrypt_question_data",
        return_value=decrypted,
    ) as decrypt_mock:
        with patch(
            "QuizGenerator.regenerate.regenerate_from_yaml_metadata",
            return_value=expected,
        ) as yaml_regen_mock:
            upload_func = Mock()
            result = regenerate_from_encrypted(
                "encrypted-token",
                points=2.5,
                yaml_text='name: "Q"\nquestions: {}\n',
                image_mode="upload",
                upload_func=upload_func,
            )

    assert result == expected
    decrypt_mock.assert_called_once_with("encrypted-token")
    yaml_regen_mock.assert_called_once_with(
        question_id="q-seeded-1",
        seed=123,
        points=2.5,
        yaml_id="yaml-alpha",
        yaml_path=None,
        yaml_text='name: "Q"\nquestions: {}\n',
        yaml_docs=None,
        image_mode="upload",
        upload_func=upload_func,
    )


def test_regenerate_from_encrypted_routes_legacy_metadata():
    decrypted = {
        "question_type": "fromtext",
        "seed": 77,
        "version": "1.0",
        "config": {"text": "hello"},
    }
    expected = {"question_type": "fromtext", "seed": 77}

    with patch(
        "QuizGenerator.regenerate.QuestionQRCode.decrypt_question_data",
        return_value=decrypted,
    ) as decrypt_mock:
        with patch(
            "QuizGenerator.regenerate.regenerate_from_metadata",
            return_value=expected,
        ) as regen_mock:
            result = regenerate_from_encrypted(
                "encrypted-token",
                points=4.0,
                image_mode="none",
            )

    assert result == expected
    decrypt_mock.assert_called_once_with("encrypted-token")
    regen_mock.assert_called_once_with(
        "fromtext",
        77,
        "1.0",
        4.0,
        {"text": "hello"},
        image_mode="none",
        upload_func=None,
    )


def test_regenerate_question_answer_requires_yaml_for_question_id():
    decrypted = {
        "question_id": "q-seeded-2",
        "yaml_id": "yaml-beta",
        "seed": 999,
    }

    with patch(
        "QuizGenerator.regenerate.QuestionQRCode.decrypt_question_data",
        return_value=decrypted,
    ):
        result = regenerate_question_answer({"q": 3, "pts": 5.0, "s": "encrypted"})

    assert result == {
        "question_number": 3,
        "points": 5.0,
        "question_id": "q-seeded-2",
        "yaml_id": "yaml-beta",
        "seed": 999,
    }


def test_regenerate_question_answer_routes_to_yaml_when_available():
    decrypted = {
        "question_id": "q-seeded-3",
        "yaml_id": "yaml-gamma",
        "seed": 101,
    }
    expected = {"answers": {"kind": "essay", "data": []}, "question_type": "fromtext"}

    with patch(
        "QuizGenerator.regenerate.QuestionQRCode.decrypt_question_data",
        return_value=decrypted,
    ):
        with patch(
            "QuizGenerator.regenerate.regenerate_from_yaml_metadata",
            return_value=expected.copy(),
        ) as yaml_regen_mock:
            result = regenerate_question_answer(
                {"q": 4, "pts": 2.0, "s": "encrypted"},
                yaml_text='name: "Q"\nquestions: {}\n',
                image_mode="inline",
            )

    assert result["question_number"] == 4
    yaml_regen_mock.assert_called_once_with(
        question_id="q-seeded-3",
        seed=101,
        points=2.0,
        yaml_id="yaml-gamma",
        yaml_path=None,
        yaml_text='name: "Q"\nquestions: {}\n',
        yaml_docs=None,
        image_mode="inline",
        upload_func=None,
    )


def test_regenerate_question_answer_missing_required_fields_returns_none():
    assert regenerate_question_answer({"q": 1}) is None
    assert regenerate_question_answer({"pts": 1.0}) is None
