#!env python
from __future__ import annotations

import abc
import difflib
import logging
import random

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
  
  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    rng = random.Random(rng_seed)
    hard_drive_rotation_speed = 100 * rng.randint(36, 150)
    seek_delay = float(round(rng.randrange(3, 20), 2))
    transfer_rate = rng.randint(50, 300)
    number_of_reads = rng.randint(1, 20)
    size_of_reads = rng.randint(1, 10)

    rotational_delay = (1 / hard_drive_rotation_speed) * (60 / 1) * (1000 / 1) * (1/2)
    access_delay = rotational_delay + seek_delay
    transfer_delay = 1000 * (size_of_reads * number_of_reads) / 1024 / transfer_rate
    disk_access_delay = access_delay * number_of_reads + transfer_delay

    return {
      "hard_drive_rotation_speed": hard_drive_rotation_speed,
      "seek_delay": seek_delay,
      "transfer_rate": transfer_rate,
      "number_of_reads": number_of_reads,
      "size_of_reads": size_of_reads,
      "rotational_delay": rotational_delay,
      "access_delay": access_delay,
      "transfer_delay": transfer_delay,
      "disk_access_delay": disk_access_delay,
    }

  @classmethod
  def _build_body(cls, context):
    """Build question body and collect answers."""
    rotational_answer = ca.AnswerTypes.Float(context["rotational_delay"])
    access_answer = ca.AnswerTypes.Float(context["access_delay"])
    transfer_answer = ca.AnswerTypes.Float(context["transfer_delay"])
    disk_access_answer = ca.AnswerTypes.Float(context["disk_access_delay"])

    # Create parameter info table using mixin
    parameter_info = {
      "Hard Drive Rotation Speed": f"{context['hard_drive_rotation_speed']}RPM",
      "Seek Delay": f"{context['seek_delay']}ms",
      "Transfer Rate": f"{context['transfer_rate']}MB/s",
      "Number of Reads": f"{context['number_of_reads']}",
      "Size of Reads": f"{context['size_of_reads']}KB"
    }

    parameter_table = cls.create_info_table(parameter_info)

    # Create answer table with multiple rows using mixin
    answer_rows = [
      {"Variable": "Rotational Delay", "Value": rotational_answer},
      {"Variable": "Access Delay", "Value": access_answer},
      {"Variable": "Transfer Delay", "Value": transfer_answer},
      {"Variable": "Total Disk Access Delay", "Value": disk_access_answer}
    ]

    answer_table = cls.create_answer_table(
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

    body = cls.create_parameter_calculation_body(
      intro_text=intro_text,
      parameter_table=parameter_table,
      answer_table=answer_table,
      additional_instructions=instructions
    )

    return body

  @classmethod
  def _build_explanation(cls, context):
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
        + f"\\frac{{1 minute}}{{{context['hard_drive_rotation_speed']}revolutions}}"
        + r"\cdot \frac{60 seconds}{1 minute} \cdot \frac{1000 ms}{1 second} \cdot \frac{1 revolution}{2} = "
        + f"{context['rotational_delay']:0.2f}ms",
      )
    ])
    
    explanation.add_elements([
      ca.Paragraph([
        "Now we can calculate:",
      ]),
      ca.Equation(
        f"t_{{access}} "
        f"= t_{{rotation}} + t_{{seek}} "
        f"= {context['rotational_delay']:0.2f}ms + {context['seek_delay']:0.2f}ms = {context['access_delay']:0.2f}ms"
      )
    ])
    
    explanation.add_elements([
      ca.Paragraph([r"Next we need to calculate our transfer delay, $t_{transfer}$, which we do as:"]),
      ca.Equation(
        f"t_{{transfer}} "
        f"= \\frac{{{context['number_of_reads']} \\cdot {context['size_of_reads']}KB}}{{1}} \\cdot \\frac{{1MB}}{{1024KB}} "
        f"\\cdot \\frac{{1 second}}{{{context['transfer_rate']}MB}} \\cdot \\frac{{1000ms}}{{1second}} "
        f"= {context['transfer_delay']:0.2}ms"
      )
    ])
    
    explanation.add_elements([
      ca.Paragraph(["Putting these together we get:"]),
      ca.Equation(
        f"t_{{total}} "
        f"= \\text{{(# reads)}} \\cdot t_{{access}} + t_{{transfer}} "
        f"= {context['number_of_reads']} \\cdot {context['access_delay']:0.2f} + {context['transfer_delay']:0.2f} "
        f"= {context['disk_access_delay']:0.2f}ms")
    ])
    return explanation


@QuestionRegistry.register()
class INodeAccesses(IOQuestion, TableQuestionMixin, BodyTemplatesMixin):
  
  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    rng = random.Random(rng_seed)
    inode_size = 2**rng.randint(6, 10)
    block_size = inode_size * rng.randint(8, 20)
    inode_number = rng.randint(0, 256)
    inode_start_location = block_size * rng.randint(2, 5)

    inode_address = inode_start_location + inode_number * inode_size
    inode_block = inode_address // block_size
    inode_address_in_block = inode_address % block_size
    inode_index_in_block = int(inode_address_in_block / inode_size)

    return {
      "inode_size": inode_size,
      "block_size": block_size,
      "inode_number": inode_number,
      "inode_start_location": inode_start_location,
      "inode_address": inode_address,
      "inode_block": inode_block,
      "inode_address_in_block": inode_address_in_block,
      "inode_index_in_block": inode_index_in_block,
    }

  @classmethod
  def _build_body(cls, context):
    """Build question body and collect answers."""
    inode_address_answer = ca.AnswerTypes.Int(context["inode_address"])
    inode_block_answer = ca.AnswerTypes.Int(context["inode_block"])
    inode_address_in_block_answer = ca.AnswerTypes.Int(context["inode_address_in_block"])
    inode_index_in_block_answer = ca.AnswerTypes.Int(context["inode_index_in_block"])

    # Create parameter info table using mixin
    parameter_info = {
      "Block Size": f"{context['block_size']} Bytes",
      "Inode Number": f"{context['inode_number']}",
      "Inode Start Location": f"{context['inode_start_location']} Bytes",
      "Inode size": f"{context['inode_size']} Bytes"
    }

    parameter_table = cls.create_info_table(parameter_info)

    # Create answer table with multiple rows using mixin
    answer_rows = [
      {"Variable": "Inode address", "Value": inode_address_answer},
      {"Variable": "Block containing inode", "Value": inode_block_answer},
      {"Variable": "Inode address (offset) within block", "Value": inode_address_in_block_answer},
      {"Variable": "Inode index within block", "Value": inode_index_in_block_answer}
    ]

    answer_table = cls.create_answer_table(
      headers=["Variable", "Value"],
      data_rows=answer_rows,
      answer_columns=["Value"]
    )

    # Use mixin to create complete body with both tables
    intro_text = "Given the information below, please calculate the following values."

    body = cls.create_parameter_calculation_body(
      intro_text=intro_text,
      parameter_table=parameter_table,
      answer_table=answer_table,
      # additional_instructions=instructions
    )

    return body

  @classmethod
  def _build_explanation(cls, context):
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
          f"{context['inode_start_location']} + {context['inode_number']} \\cdot {context['inode_size']}",
          f"{context['inode_address']}"
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
        f"{context['inode_address']} \\mathbin{{//}} {context['block_size']}",
        f"{context['inode_block']}"
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
        f"{context['inode_address']} \\bmod {context['block_size']}",
        f"{context['inode_address_in_block']}"
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
        f"\\dfrac{{{context['inode_address_in_block']}}}{{{context['inode_size']}}}",
        f"{context['inode_index_in_block']}"
      ]
    ))

    return explanation


@QuestionRegistry.register()
class VSFS_states(IOQuestion):

  from .ostep13_vsfs import fs as vsfs
  
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.answer_kind = ca.Answer.CanvasAnswerKind.MULTIPLE_DROPDOWN

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    rng = random.Random(rng_seed)
    num_steps = kwargs.get("num_steps", 10)

    fs = cls.vsfs(4, 4, rng)
    operations = fs.run_for_steps(num_steps)

    start_state = operations[-1]["start_state"]
    end_state = operations[-1]["end_state"]

    command_answer = ca.Answer.dropdown(
      f"{operations[-1]['cmd']}",
      baffles=list(set([op['cmd'] for op in operations[:-1] if op != operations[-1]['cmd']])),
      label="Command"
    )

    return {
      "start_state": start_state,
      "end_state": end_state,
      "command_answer": command_answer,
    }

  @classmethod
  def _build_body(cls, context):
    """Build question body and collect answers."""
    body = ca.Section()

    body.add_element(ca.Paragraph(["What operation happens between these two states?"]))

    body.add_element(
      ca.Code(
        context["start_state"],
        make_small=True
      )
    )

    body.add_element(ca.AnswerBlock(context["command_answer"]))

    body.add_element(
      ca.Code(
        context["end_state"],
        make_small=True
      )
    )

    return body

  @classmethod
  def _build_explanation(cls, context):
    explanation = ca.Section()
    
    log.debug(f"start_state: {context['start_state']}")
    log.debug(f"end_state: {context['end_state']}")
    
    explanation.add_elements([
      ca.Paragraph([
        "The key thing to pay attention to when solving these problems is where there are differences between the start state and the end state.",
        "In this particular problem, we can see that these lines are different:"
      ])
    ])
    
    chunk_to_add = []
    lines_that_changed = []
    for start_line, end_line in zip(context["start_state"].split('\n'), context["end_state"].split('\n')):
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
        highlight_changes(context["start_state"], context["end_state"])
      )
    )

    return explanation
