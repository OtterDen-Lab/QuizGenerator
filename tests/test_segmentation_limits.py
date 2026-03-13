import pytest

from QuizGenerator.generation.premade_questions.cst334.memory_questions import (
  Segmentation,
)


def test_segmentation_respects_configured_bit_caps():
  for seed in range(20):
    context = Segmentation._build_context(
      rng_seed=seed,
      max_bits=12,
      max_virtual_bits=7,
    )
    assert context["virtual_bits"] <= 7
    assert context["physical_bits"] <= 12
    assert context["virtual_bits"] < context["physical_bits"]


def test_segmentation_accepts_explicit_physical_cap_alias():
  context = Segmentation._build_context(
    rng_seed=123,
    max_physical_bits=11,
    max_virtual_bits=6,
  )
  assert context["virtual_bits"] <= 6
  assert context["physical_bits"] <= 11


def test_segmentation_derives_virtual_cap_from_max_bits_when_unspecified():
  context = Segmentation._build_context(
    rng_seed=123,
    max_bits=8,
  )
  assert context["virtual_bits"] <= 7
  assert context["physical_bits"] <= 8


def test_segmentation_rejects_impossible_bit_caps():
  with pytest.raises(ValueError, match="max_bits/max_physical_bits"):
    Segmentation._build_context(rng_seed=1, max_bits=5)

  with pytest.raises(ValueError, match="max_virtual_bits"):
    Segmentation._build_context(rng_seed=1, max_virtual_bits=4)

  with pytest.raises(ValueError, match="smaller than"):
    Segmentation._build_context(rng_seed=1, max_bits=8, max_virtual_bits=8)
