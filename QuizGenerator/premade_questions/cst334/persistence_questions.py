#!env python
from __future__ import annotations

import abc
import difflib
import logging

from QuizGenerator.question import Question, QuestionRegistry
import QuizGenerator.contentast as ca
from QuizGenerator.mixins import TableQuestionMixin, BodyTemplatesMixin

log = logging.getLogger(__name__)


class IOQuestion(Question, abc.ABC):
  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.IO)
    super().__init__(*args, **kwargs)
  

@QuestionRegistry.register()
class HardDriveAccessTime(IOQuestion, TableQuestionMixin, BodyTemplatesMixin):
  
  def refresh(self, *args, **kwargs):
    super().refresh(*args, **kwargs)
    
    self.hard_drive_rotation_speed = 100 * self.rng.randint(36, 150)  # e.g. 3600rpm to 15000rpm
    self.seek_delay = float(round(self.rng.randrange(3, 20), 2))
    self.transfer_rate = self.rng.randint(50, 300)
    self.number_of_reads = self.rng.randint(1, 20)
    self.size_of_reads = self.rng.randint(1, 10)
    
    self.rotational_delay = (1 / self.hard_drive_rotation_speed) * (60 / 1) * (1000 / 1) * (1/2)
    self.access_delay = self.rotational_delay + self.seek_delay
    self.transfer_delay = 1000 * (self.size_of_reads * self.number_of_reads) / 1024 / self.transfer_rate
    self.disk_access_delay = self.access_delay * self.number_of_reads + self.transfer_delay
    
    self.answers.update({
      "answer__rotational_delay"  : ca.AnswerTypes.Float(self.rotational_delay),
      "answer__access_delay"      : ca.AnswerTypes.Float(self.access_delay),
      "answer__transfer_delay"    : ca.AnswerTypes.Float(self.transfer_delay),
      "answer__disk_access_delay" : ca.AnswerTypes.Float(self.disk_access_delay),
    })
  
  def _get_body(self, *args, **kwargs):
    """Build question body and collect answers."""
    answers = [
      self.answers["answer__rotational_delay"],
      self.answers["answer__access_delay"],
      self.answers["answer__transfer_delay"],
      self.answers["answer__disk_access_delay"],
    ]

    # Create parameter info table using mixin
    parameter_info = {
      "Hard Drive Rotation Speed": f"{self.hard_drive_rotation_speed}RPM",
      "Seek Delay": f"{self.seek_delay}ms",
      "Transfer Rate": f"{self.transfer_rate}MB/s",
      "Number of Reads": f"{self.number_of_reads}",
      "Size of Reads": f"{self.size_of_reads}KB"
    }

    parameter_table = self.create_info_table(parameter_info)

    # Create answer table with multiple rows using mixin
    answer_rows = [
      {"Variable": "Rotational Delay", "Value": "answer__rotational_delay"},
      {"Variable": "Access Delay", "Value": "answer__access_delay"},
      {"Variable": "Transfer Delay", "Value": "answer__transfer_delay"},
      {"Variable": "Total Disk Access Delay", "Value": "answer__disk_access_delay"}
    ]

    answer_table = self.create_answer_table(
      headers=["Variable", "Value"],
      data_rows=answer_rows,
      answer_columns=["Value"]
    )

    # Use mixin to create complete body with both tables
    intro_text = "Given the information below, please calculate the following values."

    instructions = (
      f"Make sure that if you round your answers you use the unrounded values for your final calculations, "
      f"otherwise you may introduce error into your calculations."
      f"(i.e. don't use your rounded answers to calculate your overall answer)"
    )

    body = self.create_parameter_calculation_body(
      intro_text=intro_text,
      parameter_table=parameter_table,
      answer_table=answer_table,
      additional_instructions=instructions
    )

    return body, answers

  def get_body(self, *args, **kwargs) -> ca.Section:
    """Build question body (backward compatible interface)."""
    body, _ = self._get_body(*args, **kwargs)
    return body

  def _get_explanation(self):
    explanation = ca.Section()
    
    explanation.add_element(
      ca.Paragraph([
        "To calculate the total disk access time (or \"delay\"), "
        "we should first calculate each of the individual parts.",
        r"Since we know that  $t_{total} = (\text{# of reads}) \cdot t_{access} + t_{transfer}$"
        r"we therefore need to calculate $t_{access}$ and  $t_{transfer}$, where "
        r"$t_{access} = t_{rotation} + t_{seek}$.",
      ])
    )
    
    explanation.add_elements([
      ca.Paragraph(["Starting with the rotation delay, we calculate:"]),
      ca.Equation(
        "t_{rotation} = "
        + f"\\frac{{1 minute}}{{{self.hard_drive_rotation_speed}revolutions}}"
        + r"\cdot \frac{60 seconds}{1 minute} \cdot \frac{1000 ms}{1 second} \cdot \frac{1 revolution}{2} = "
        + f"{self.rotational_delay:0.2f}ms",
      )
    ])
    
    explanation.add_elements([
      ca.Paragraph([
        "Now we can calculate:",
      ]),
      ca.Equation(
        f"t_{{access}} "
        f"= t_{{rotation}} + t_{{seek}} "
        f"= {self.rotational_delay:0.2f}ms + {self.seek_delay:0.2f}ms = {self.access_delay:0.2f}ms"
      )
    ])
    
    explanation.add_elements([
      ca.Paragraph([r"Next we need to calculate our transfer delay, $t_{transfer}$, which we do as:"]),
      ca.Equation(
        f"t_{{transfer}} "
        f"= \\frac{{{self.number_of_reads} \\cdot {self.size_of_reads}KB}}{{1}} \\cdot \\frac{{1MB}}{{1024KB}} "
        f"\\cdot \\frac{{1 second}}{{{self.transfer_rate}MB}} \\cdot \\frac{{1000ms}}{{1second}} "
        f"= {self.transfer_delay:0.2}ms"
      )
    ])
    
    explanation.add_elements([
      ca.Paragraph(["Putting these together we get:"]),
      ca.Equation(
        f"t_{{total}} "
        f"= \\text{{(# reads)}} \\cdot t_{{access}} + t_{{transfer}} "
        f"= {self.number_of_reads} \\cdot {self.access_delay:0.2f} + {self.transfer_delay:0.2f} "
        f"= {self.disk_access_delay:0.2f}ms")
    ])
    return explanation, []

  def get_explanation(self) -> ca.Section:
    """Build question explanation (backward compatible interface)."""
    explanation, _ = self._get_explanation()
    return explanation


@QuestionRegistry.register()
class INodeAccesses(IOQuestion, TableQuestionMixin, BodyTemplatesMixin):
  
  def refresh(self, *args, **kwargs):
    super().refresh(*args, **kwargs)
    
    # Calculating this first to use blocksize as an even multiple of it
    self.inode_size = 2**self.rng.randint(6, 10)
    
    self.block_size = self.inode_size * self.rng.randint(8, 20)
    self.inode_number = self.rng.randint(0, 256)
    self.inode_start_location = self.block_size * self.rng.randint(2, 5)
    
    self.inode_address = self.inode_start_location + self.inode_number * self.inode_size
    self.inode_block = self.inode_address // self.block_size
    self.inode_address_in_block = self.inode_address % self.block_size
    self.inode_index_in_block = int(self.inode_address_in_block / self.inode_size)
    
    self.answers.update({
      "answer__inode_address": ca.AnswerTypes.Int(self.inode_address),
      "answer__inode_block": ca.AnswerTypes.Int(self.inode_block),
      "answer__inode_address_in_block": ca.AnswerTypes.Int(self.inode_address_in_block),
      "answer__inode_index_in_block": ca.AnswerTypes.Int(self.inode_index_in_block),
    })
  
  def _get_body(self):
    """Build question body and collect answers."""
    answers = [
      self.answers["answer__inode_address"],
      self.answers["answer__inode_block"],
      self.answers["answer__inode_address_in_block"],
      self.answers["answer__inode_index_in_block"],
    ]

    # Create parameter info table using mixin
    parameter_info = {
      "Block Size": f"{self.block_size} Bytes",
      "Inode Number": f"{self.inode_number}",
      "Inode Start Location": f"{self.inode_start_location} Bytes",
      "Inode size": f"{self.inode_size} Bytes"
    }

    parameter_table = self.create_info_table(parameter_info)

    # Create answer table with multiple rows using mixin
    answer_rows = [
      {"Variable": "Inode address", "Value": "answer__inode_address"},
      {"Variable": "Block containing inode", "Value": "answer__inode_block"},
      {"Variable": "Inode address (offset) within block", "Value": "answer__inode_address_in_block"},
      {"Variable": "Inode index within block", "Value": "answer__inode_index_in_block"}
    ]

    answer_table = self.create_answer_table(
      headers=["Variable", "Value"],
      data_rows=answer_rows,
      answer_columns=["Value"]
    )

    # Use mixin to create complete body with both tables
    intro_text = "Given the information below, please calculate the following values."

    body = self.create_parameter_calculation_body(
      intro_text=intro_text,
      parameter_table=parameter_table,
      answer_table=answer_table,
      # additional_instructions=instructions
    )

    return body, answers

  def get_body(self) -> ca.Section:
    """Build question body (backward compatible interface)."""
    body, _ = self._get_body()
    return body

  def _get_explanation(self):
    explanation = ca.Section()
    
    explanation.add_element(
      ca.Paragraph([
        "If we are given an inode number, there are a few steps that we need to take to load the actual inode.  "
        "These consist of determining the address of the inode, which block would contain it, "
        "and then its address within the block.",
        "To find the inode address, we calculate:",
      ])
    )
    
    explanation.add_element(
      ca.Equation.make_block_equation__multiline_equals(
        r"(\text{Inode address})",
        [
          r"(\text{Inode Start Location}) + (\text{inode #}) \cdot (\text{inode size})",
          f"{self.inode_start_location} + {self.inode_number} \\cdot {self.inode_size}",
          f"{self.inode_address}"
        ])
    )
    
    explanation.add_element(
      ca.Paragraph([
        "Next, we us this to figure out what block the inode is in.  "
        "We do this directly so we know what block to load, "
        "thus minimizing the number of loads we have to make.",
      ])
    )
    explanation.add_element(ca.Equation.make_block_equation__multiline_equals(
      r"\text{Block containing inode}",
      [
        r"(\text{Inode address}) \mathbin{//} (\text{block size})",
        f"{self.inode_address} \\mathbin{{//}} {self.block_size}",
        f"{self.inode_block}"
      ]
    ))
    
    explanation.add_element(
      ca.Paragraph([
        "When we load this block, we now have in our system memory "
        "(remember, blocks on the hard drive are effectively useless to us until they're in main memory!), "
        "the inode, so next we need to figure out where it is within that block."
        "This means that we'll need to find the offset into this block.  "
        "We'll calculate this both as the offset in bytes, and also in number of inodes, "
        "since we can use array indexing.",
      ])
    )
    
    explanation.add_element(ca.Equation.make_block_equation__multiline_equals(
      r"\text{offset within block}",
      [
        r"(\text{Inode address}) \bmod (\text{block size})",
        f"{self.inode_address} \\bmod {self.block_size}",
        f"{self.inode_address_in_block}"
      ]
    ))
    
    explanation.add_element(
      ca.Text("Remember that `mod` is the same as `%`, the modulo operation.")
    )
    
    explanation.add_element(ca.Paragraph(["and"]))
      
    explanation.add_element(ca.Equation.make_block_equation__multiline_equals(
      r"\text{index within block}",
      [
        r"\dfrac{\text{offset within block}}{\text{inode size}}",
        f"\\dfrac{{{self.inode_address_in_block}}}{{{self.inode_size}}}",
        f"{self.inode_index_in_block}"
      ]
    ))

    return explanation, []

  def get_explanation(self) -> ca.Section:
    """Build question explanation (backward compatible interface)."""
    explanation, _ = self._get_explanation()
    return explanation


@QuestionRegistry.register()
class VSFS_states(IOQuestion):

  from .ostep13_vsfs import fs as vsfs
  
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.answer_kind = ca.Answer.CanvasAnswerKind.MULTIPLE_DROPDOWN
    
    self.num_steps = kwargs.get("num_steps", 10)
  
  def refresh(self, *args, **kwargs):
    super().refresh(*args, **kwargs)
    
    fs = self.vsfs(4, 4, self.rng)
    operations = fs.run_for_steps(self.num_steps)
    
    self.start_state = operations[-1]["start_state"]
    self.end_state = operations[-1]["end_state"]
    
    wrong_answers = list(filter(
      lambda o: o != operations[-1]["cmd"],
      map(
        lambda o: o["cmd"],
        operations
      )
    ))
    self.rng.shuffle(wrong_answers)
    
    self.answers["answer__cmd"] = ca.Answer.dropdown(
      f"{operations[-1]['cmd']}",
      baffles=list(set([op['cmd'] for op in operations[:-1] if op != operations[-1]['cmd']])),
      label="Command"
    )
  
  def _get_body(self):
    """Build question body and collect answers."""
    answers = [self.answers["answer__cmd"]]

    body = ca.Section()

    body.add_element(ca.Paragraph(["What operation happens between these two states?"]))

    body.add_element(
      ca.Code(
        self.start_state,
        make_small=True
      )
    )

    body.add_element(ca.AnswerBlock(self.answers["answer__cmd"]))

    body.add_element(
      ca.Code(
        self.end_state,
        make_small=True
      )
    )

    return body, answers

  def get_body(self) -> ca.Section:
    """Build question body (backward compatible interface)."""
    body, _ = self._get_body()
    return body

  def _get_explanation(self):
    explanation = ca.Section()
    
    log.debug(f"self.start_state: {self.start_state}")
    log.debug(f"self.end_state: {self.end_state}")
    
    explanation.add_elements([
      ca.Paragraph([
        "The key thing to pay attention to when solving these problems is where there are differences between the start state and the end state.",
        "In this particular problem, we can see that these lines are different:"
      ])
    ])
    
    chunk_to_add = []
    lines_that_changed = []
    for start_line, end_line in zip(self.start_state.split('\n'), self.end_state.split('\n')):
      if start_line == end_line:
        continue
      lines_that_changed.append((start_line, end_line))
      chunk_to_add.append(
        f" - `{start_line}` -> `{end_line}`"
      )
    
    explanation.add_element(
      ca.Paragraph(chunk_to_add)
    )
    
    chunk_to_add = [
      "A great place to start is to check to see if the bitmaps have changed as this can quickly tell us a lot of information"
    ]
    
    inode_bitmap_lines = list(filter(lambda s: "inode bitmap" in s[0], lines_that_changed))
    data_bitmap_lines = list(filter(lambda s: "data bitmap" in s[0], lines_that_changed))
    
    def get_bitmap(line: str) -> str:
      log.debug(f"line: {line}")
      return line.split()[-1]
    
    def highlight_changes(a: str, b: str) -> str:
      matcher = difflib.SequenceMatcher(None, a, b)
      result = []
      
      for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
          result.append(b[j1:j2])
        elif tag in ("insert", "replace"):
          result.append(f"***{b[j1:j2]}***")
        # for "delete", do nothing since text is removed
      
      return "".join(result)
    
    if len(inode_bitmap_lines) > 0:
      inode_bitmap_lines = inode_bitmap_lines[0]
      chunk_to_add.append(f"The inode bitmap lines have changed from {get_bitmap(inode_bitmap_lines[0])} to {get_bitmap(inode_bitmap_lines[1])}.")
      if get_bitmap(inode_bitmap_lines[0]).count('1') < get_bitmap(inode_bitmap_lines[1]).count('1'):
        chunk_to_add.append("We can see that we have added an inode, so we have either called `creat` or `mkdir`.")
      else:
        chunk_to_add.append("We can see that we have removed an inode, so we have called `unlink`.")
    
    if len(data_bitmap_lines) > 0:
      data_bitmap_lines = data_bitmap_lines[0]
      chunk_to_add.append(f"The inode bitmap lines have changed from {get_bitmap(data_bitmap_lines[0])} to {get_bitmap(data_bitmap_lines[1])}.")
      if get_bitmap(data_bitmap_lines[0]).count('1') < get_bitmap(data_bitmap_lines[1]).count('1'):
        chunk_to_add.append("We can see that we have added a data block, so we have either called `mkdir` or `write`.")
      else:
        chunk_to_add.append("We can see that we have removed a data block, so we have `unlink`ed a file.")
    
    if len(data_bitmap_lines) == 0 and len(inode_bitmap_lines) == 0:
      chunk_to_add.append("If they have not changed, then we know we must have eithered called `link` or `unlink` and must check the references.")
      
    explanation.add_element(
      ca.Paragraph(chunk_to_add)
    )
    
    explanation.add_elements([
      ca.Paragraph(["The overall changes are highlighted with `*` symbols below"])
    ])
    
    explanation.add_element(
      ca.Code(
        highlight_changes(self.start_state, self.end_state)
      )
    )

    return explanation, []

  def get_explanation(self) -> ca.Section:
    """Build question explanation (backward compatible interface)."""
    explanation, _ = self._get_explanation()
    return explanation
  