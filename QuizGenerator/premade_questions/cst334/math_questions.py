#!/usr/bin/env python
import abc
import logging
import math
import random

import QuizGenerator.contentast as ca
from QuizGenerator.constants import MathRanges
from QuizGenerator.question import Question, QuestionRegistry

log = logging.getLogger(__name__)


class MathQuestion(Question, abc.ABC):
  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.MATH)
    super().__init__(*args, **kwargs)


@QuestionRegistry.register()
class BitsAndBytes(MathQuestion):
  
  MIN_BITS = MathRanges.DEFAULT_MIN_MATH_BITS
  MAX_BITS = MathRanges.DEFAULT_MAX_MATH_BITS
  
  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    rng = random.Random(rng_seed)
    from_binary = (0 == rng.randint(0, 1))
    num_bits = rng.randint(cls.MIN_BITS, cls.MAX_BITS)
    num_bytes = int(math.pow(2, num_bits))
    return {
      "from_binary": from_binary,
      "num_bits": num_bits,
      "num_bytes": num_bytes,
    }

  @classmethod
  def _build_body(cls, context):
    """Build question body and collect answers."""
    if context["from_binary"]:
      answer = ca.AnswerTypes.Int(
        context["num_bytes"],
        label="Address space size",
        unit="Bytes"
      )
    else:
      answer = ca.AnswerTypes.Int(
        context["num_bits"],
        label="Number of bits in address",
        unit="bits"
      )

    body = ca.Section()
    body.add_element(
      ca.Paragraph([
        f"Given that we have "
        f"{context['num_bits'] if context['from_binary'] else context['num_bytes']} "
        f"{'bits' if context['from_binary'] else 'bytes'}, "
        f"how many {'bits' if not context['from_binary'] else 'bytes'} "
        f"{'do we need to address our memory' if not context['from_binary'] else 'of memory can be addressed'}?"
      ])
    )

    body.add_element(ca.AnswerBlock(answer))

    return body

  @classmethod
  def _build_explanation(cls, context):
    explanation = ca.Section()
    
    explanation.add_element(
      ca.Paragraph([
        "Remember that for these problems we use one of these two equations (which are equivalent)"
      ])
    )
    explanation.add_elements([
      ca.Equation(r"log_{2}(\text{#bytes}) = \text{#bits}"),
      ca.Equation(r"2^{(\text{#bits})} = \text{#bytes}")
    ])
    
    explanation.add_element(
      ca.Paragraph(["Therefore, we calculate:"])
    )
    
    if context["from_binary"]:
      explanation.add_element(
        ca.Equation(
          f"2 ^ {{{context['num_bits']}bits}} = \\textbf{{{context['num_bytes']}}}\\text{{bytes}}"
        )
      )
    else:
      explanation.add_element(
        ca.Equation(
          f"log_{{2}}({context['num_bytes']} \\text{{bytes}}) = \\textbf{{{context['num_bits']}}}\\text{{bits}}"
        )
      )

    return explanation


@QuestionRegistry.register()
class HexAndBinary(MathQuestion):
  
  MIN_HEXITS = 1
  MAX_HEXITS = 8
  
  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    rng = random.Random(rng_seed)
    from_binary = rng.choice([True, False])
    number_of_hexits = rng.randint(1, 8)
    value = rng.randint(1, 16**number_of_hexits)
    hex_val = f"0x{value:0{number_of_hexits}X}"
    binary_val = f"0b{value:0{4*number_of_hexits}b}"
    return {
      "from_binary": from_binary,
      "number_of_hexits": number_of_hexits,
      "value": value,
      "hex_val": hex_val,
      "binary_val": binary_val,
    }

  @classmethod
  def _build_body(cls, context):
    """Build question body and collect answers."""
    if context["from_binary"]:
      answer = ca.AnswerTypes.String(context["hex_val"], label="Value in hex")
    else:
      answer = ca.AnswerTypes.String(context["binary_val"], label="Value in binary")

    body = ca.Section()

    body.add_element(
      ca.Paragraph([
        f"Given the number {context['hex_val'] if not context['from_binary'] else context['binary_val']} "
        f"please convert it to {'hex' if context['from_binary'] else 'binary'}.",
        "Please include base indicator all padding zeros as appropriate (e.g. 0x01 should be 0b00000001)",
      ])
    )

    body.add_element(ca.AnswerBlock(answer))

    return body

  @classmethod
  def _build_explanation(cls, context):
    explanation = ca.Section()
    
    paragraph = ca.Paragraph([
      "The core idea for converting between binary and hex is to divide and conquer.  "
      "Specifically, each hexit (hexadecimal digit) is equivalent to 4 bits.  "
    ])
    
    if context["from_binary"]:
      paragraph.add_line(
        "Therefore, we need to consider each group of 4 bits together and convert them to the appropriate hexit."
      )
    else:
      paragraph.add_line(
        "Therefore, we need to consider each hexit and convert it to the appropriate 4 bits."
      )
    
    explanation.add_element(paragraph)
    
    # Generate translation table
    binary_str = f"{context['value']:0{4*context['number_of_hexits']}b}"
    hex_str = f"{context['value']:0{context['number_of_hexits']}X}"
    
    explanation.add_element(
      ca.Table(
        data=[
          ["0b"] + [binary_str[i:i+4] for i in range(0, len(binary_str), 4)],
          ["0x"] + list(hex_str)
        ],
        # alignments='center', #['center' for _ in range(0, 1+len(hex_str))],
        padding=False
        
      )
    )
    
    if context["from_binary"]:
      explanation.add_element(
        ca.Paragraph([
        f"Which gives us our hex value of: 0x{hex_str}"
        ])
      )
    else:
      explanation.add_element(
        ca.Paragraph([
          f"Which gives us our binary value of: 0b{binary_str}"
        ])
      )

    return explanation


@QuestionRegistry.register()
class AverageMemoryAccessTime(MathQuestion):
  
  CHANCE_OF_99TH_PERCENTILE = 0.75
  
  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    rng = random.Random(rng_seed)
    orders_of_magnitude_different = rng.randint(1, 4)
    hit_latency = rng.randint(1, 9)
    miss_latency = int(rng.randint(1, 9) * math.pow(10, orders_of_magnitude_different))

    if rng.random() < cls.CHANCE_OF_99TH_PERCENTILE:
      hit_rate = (99 + rng.random()) / 100
    else:
      hit_rate = rng.random()

    hit_rate = round(hit_rate, 4)
    amat = hit_rate * hit_latency + (1 - hit_rate) * miss_latency
    show_miss_rate = rng.random() > 0.5

    return {
      "hit_latency": hit_latency,
      "miss_latency": miss_latency,
      "hit_rate": hit_rate,
      "amat": amat,
      "show_miss_rate": show_miss_rate,
    }

  @classmethod
  def _build_body(cls, context):
    """Build question body and collect answers."""
    answer = ca.AnswerTypes.Float(
      context["amat"],
      label="Average Memory Access Time",
      unit="cycles"
    )

    body = ca.Section()

    # Add in background information
    body.add_element(
      ca.Paragraph([
        ca.Text("Please calculate the Average Memory Access Time given the below information. "),
        ca.Text(
          f"Please round your answer to {ca.Answer.DEFAULT_ROUNDING_DIGITS} decimal points. ",
          hide_from_latex=True
        )
      ])
    )
    table_data = [
      ["Hit Latency", f"{context['hit_latency']} cycles"],
      ["Miss Latency", f"{context['miss_latency']} cycles"]
    ]

    # Add in either miss rate or hit rate -- we only need one of them
    if context["show_miss_rate"]:
      table_data.append(["Miss Rate", f"{100 * (1 - context['hit_rate']): 0.2f}%"])
    else:
      table_data.append(["Hit Rate", f"{100 * context['hit_rate']: 0.2f}%"])

    body.add_element(
      ca.Table(
        data=table_data
      )
    )

    body.add_element(ca.LineBreak())

    body.add_element(ca.AnswerBlock(answer))

    return body

  @classmethod
  def _build_explanation(cls, context):
    explanation = ca.Section()
    
    # Add in General explanation
    explanation.add_element(
      ca.Paragraph([
        "Remember that to calculate the Average Memory Access Time "
        "we weight both the hit and miss times by their relative likelihood.",
        "That is, we calculate:"
      ])
    )
    
    # Add in equations
    explanation.add_element(
      ca.Equation.make_block_equation__multiline_equals(
        lhs="AMAT",
        rhs=[
          r"(hit\_rate)*(hit\_cost) + (1 - hit\_rate)*(miss\_cost)",
          f"({context['hit_rate']: 0.{ca.Answer.DEFAULT_ROUNDING_DIGITS}f})*({context['hit_latency']}) + "
          f"({1 - context['hit_rate']: 0.{ca.Answer.DEFAULT_ROUNDING_DIGITS}f})*({context['miss_latency']}) = "
          f"{context['amat']: 0.{ca.Answer.DEFAULT_ROUNDING_DIGITS}f}\\text{{cycles}}"
        ]
      )
    )

    return explanation
