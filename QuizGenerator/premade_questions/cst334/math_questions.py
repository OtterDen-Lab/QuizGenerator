#!env python
import abc
import logging
import math

from QuizGenerator.question import Question, QuestionRegistry
import QuizGenerator.contentast as ca
from QuizGenerator.constants import MathRanges

log = logging.getLogger(__name__)


class MathQuestion(Question, abc.ABC):
  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.MATH)
    super().__init__(*args, **kwargs)


@QuestionRegistry.register()
class BitsAndBytes(MathQuestion):
  
  MIN_BITS = MathRanges.DEFAULT_MIN_MATH_BITS
  MAX_BITS = MathRanges.DEFAULT_MAX_MATH_BITS
  
  def refresh(self, *args, **kwargs):
    super().refresh(*args, **kwargs)
    
    # Generate the important parts of the problem
    self.from_binary = (0 == self.rng.randint(0,1))
    self.num_bits = self.rng.randint(self.MIN_BITS, self.MAX_BITS)
    self.num_bytes = int(math.pow(2, self.num_bits))
    
    if self.from_binary:
      self.answers = {"answer": ca.AnswerTypes.Int(self.num_bytes,
                                               label="Address space size", unit="Bytes")}
    else:
      self.answers = {"answer": ca.AnswerTypes.Int(self.num_bits,
                                               label="Number of bits in address", unit="bits")}
  
  def _get_body(self, **kwargs):
    """Build question body and collect answers."""
    answers = [self.answers['answer']]

    body = ca.Section()
    body.add_element(
      ca.Paragraph([
        f"Given that we have "
        f"{self.num_bits if self.from_binary else self.num_bytes} {'bits' if self.from_binary else 'bytes'}, "
        f"how many {'bits' if not self.from_binary else 'bytes'} "
        f"{'do we need to address our memory' if not self.from_binary else 'of memory can be addressed'}?"
      ])
    )

    body.add_element(ca.AnswerBlock(self.answers['answer']))

    return body, answers

  def get_body(self, **kwargs) -> ca.Section:
    """Build question body (backward compatible interface)."""
    body, _ = self._get_body(**kwargs)
    return body

  def _get_explanation(self, **kwargs):
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
    
    if self.from_binary:
      explanation.add_element(
        ca.Equation(f"2 ^ {{{self.num_bits}bits}} = \\textbf{{{self.num_bytes}}}\\text{{bytes}}")
      )
    else:
      explanation.add_element(
        ca.Equation(f"log_{{2}}({self.num_bytes} \\text{{bytes}}) = \\textbf{{{self.num_bits}}}\\text{{bits}}")
      )

    return explanation, []

  def get_explanation(self, **kwargs) -> ca.Section:
    """Build question explanation (backward compatible interface)."""
    explanation, _ = self._get_explanation(**kwargs)
    return explanation


@QuestionRegistry.register()
class HexAndBinary(MathQuestion):
  
  MIN_HEXITS = 1
  MAX_HEXITS = 8
  
  def refresh(self, **kwargs):
    super().refresh(**kwargs)
    
    self.from_binary = self.rng.choice([True, False])
    self.number_of_hexits = self.rng.randint(1, 8)
    self.value = self.rng.randint(1, 16**self.number_of_hexits)
    
    self.hex_val = f"0x{self.value:0{self.number_of_hexits}X}"
    self.binary_val = f"0b{self.value:0{4*self.number_of_hexits}b}"
    
    if self.from_binary:
      self.answers['answer'] = ca.AnswerTypes.String(self.hex_val,
                                             label="Value in hex")
    else:
      self.answers['answer'] = ca.AnswerTypes.String(self.binary_val,
                                             label="Value in binary")
  
  def _get_body(self, **kwargs):
    """Build question body and collect answers."""
    answers = [self.answers['answer']]

    body = ca.Section()

    body.add_element(
      ca.Paragraph([
        f"Given the number {self.hex_val if not self.from_binary else self.binary_val} "
        f"please convert it to {'hex' if self.from_binary else 'binary'}.",
        "Please include base indicator all padding zeros as appropriate (e.g. 0x01 should be 0b00000001)",
      ])
    )

    body.add_element(ca.AnswerBlock(self.answers['answer']))

    return body, answers

  def get_body(self, **kwargs) -> ca.Section:
    """Build question body (backward compatible interface)."""
    body, _ = self._get_body(**kwargs)
    return body

  def _get_explanation(self, **kwargs):
    explanation = ca.Section()
    
    paragraph = ca.Paragraph([
      "The core idea for converting between binary and hex is to divide and conquer.  "
      "Specifically, each hexit (hexadecimal digit) is equivalent to 4 bits.  "
    ])
    
    if self.from_binary:
      paragraph.add_line(
        "Therefore, we need to consider each group of 4 bits together and convert them to the appropriate hexit."
      )
    else:
      paragraph.add_line(
        "Therefore, we need to consider each hexit and convert it to the appropriate 4 bits."
      )
    
    explanation.add_element(paragraph)
    
    # Generate translation table
    binary_str = f"{self.value:0{4*self.number_of_hexits}b}"
    hex_str = f"{self.value:0{self.number_of_hexits}X}"
    
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
    
    if self.from_binary:
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

    return explanation, []

  def get_explanation(self, **kwargs) -> ca.Section:
    """Build question explanation (backward compatible interface)."""
    explanation, _ = self._get_explanation(**kwargs)
    return explanation


@QuestionRegistry.register()
class AverageMemoryAccessTime(MathQuestion):
  
  CHANCE_OF_99TH_PERCENTILE = 0.75
  
  def refresh(self, rng_seed=None, *args, **kwargs):
    super().refresh(rng_seed=rng_seed, *args, **kwargs)
    
    # Figure out how many orders of magnitude different we are
    orders_of_magnitude_different = self.rng.randint(1,4)
    self.hit_latency = self.rng.randint(1,9)
    self.miss_latency = int(self.rng.randint(1, 9) * math.pow(10, orders_of_magnitude_different))
    
    # Add in a complication of making it sometimes very, very close
    if self.rng.random() < self.CHANCE_OF_99TH_PERCENTILE:
      # Then let's make it very close to 99%
      self.hit_rate = (99 + self.rng.random()) / 100
    else:
      self.hit_rate = self.rng.random()
      
    # Calculate the hit rate
    self.hit_rate = round(self.hit_rate, 4)
    
    # Calculate the AverageMemoryAccessTime (which is the answer itself)
    self.amat = self.hit_rate * self.hit_latency + (1 - self.hit_rate) * self.miss_latency
    
    self.answers = {
      "amat": ca.AnswerTypes.Float(self.amat, label="Average Memory Access Time", unit="cycles")
    }
    
    # Finally, do the self.rngizing of the question, to avoid these being non-deterministic
    self.show_miss_rate = self.rng.random() > 0.5
    
    # At this point, everything in the question should be set.
    pass
  
  def _get_body(self, **kwargs):
    """Build question body and collect answers."""
    answers = [self.answers["amat"]]

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
      ["Hit Latency", f"{self.hit_latency} cycles"],
      ["Miss Latency", f"{self.miss_latency} cycles"]
    ]

    # Add in either miss rate or hit rate -- we only need one of them
    if self.show_miss_rate:
      table_data.append(["Miss Rate", f"{100 * (1 - self.hit_rate): 0.2f}%"])
    else:
      table_data.append(["Hit Rate", f"{100 * self.hit_rate: 0.2f}%"])

    body.add_element(
      ca.Table(
        data=table_data
      )
    )

    body.add_element(ca.LineBreak())

    body.add_element(ca.AnswerBlock(self.answers["amat"]))

    return body, answers

  def get_body(self, **kwargs) -> ca.Section:
    """Build question body (backward compatible interface)."""
    body, _ = self._get_body(**kwargs)
    return body

  def _get_explanation(self, **kwargs):
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
          f"({self.hit_rate: 0.{ca.Answer.DEFAULT_ROUNDING_DIGITS}f})*({self.hit_latency}) + ({1 - self.hit_rate: 0.{ca.Answer.DEFAULT_ROUNDING_DIGITS}f})*({self.miss_latency}) = {self.amat: 0.{ca.Answer.DEFAULT_ROUNDING_DIGITS}f}\\text{{cycles}}"
        ]
      )
    )

    return explanation, []

  def get_explanation(self, **kwargs) -> ca.Section:
    """Build question explanation (backward compatible interface)."""
    explanation, _ = self._get_explanation(**kwargs)
    return explanation
  