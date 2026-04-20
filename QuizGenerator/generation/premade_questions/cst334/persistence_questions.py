#!/usr/bin/env python
from __future__ import annotations

import abc
import difflib
import logging
import re
import random
from fractions import Fraction

import QuizGenerator.generation.contentast as ca
from QuizGenerator.generation.mixins import BodyTemplatesMixin, TableQuestionMixin
from QuizGenerator.generation.question import Question, QuestionRegistry

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

    rotational_delay = Fraction(1, hard_drive_rotation_speed) * 60 * 1000 * Fraction(1, 2)
    access_delay = rotational_delay + Fraction(str(seek_delay))
    transfer_delay = Fraction(1000 * size_of_reads * number_of_reads, 1024 * transfer_rate)
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
    rotational_answer = ca.AnswerTypes.Float(
      context["rotational_delay"],
      display=f"{float(context['rotational_delay']):0.2f}",
      unit="ms",
    )
    access_answer = ca.AnswerTypes.Float(
      context["access_delay"],
      display=f"{float(context['access_delay']):0.2f}",
      unit="ms",
    )
    transfer_answer = ca.AnswerTypes.Float(
      context["transfer_delay"],
      display=f"{float(context['transfer_delay']):0.2f}",
      unit="ms",
    )
    disk_access_answer = ca.AnswerTypes.Float(
      context["disk_access_delay"],
      display=f"{float(context['disk_access_delay']):0.2f}",
      unit="ms",
    )

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
      {"Variable": "Rotational Delay (ms)", "Value": rotational_answer},
      {"Variable": "Access Delay (ms)", "Value": access_answer},
      {"Variable": "Transfer Delay (ms)", "Value": transfer_answer},
      {"Variable": "Total Disk Access Delay (ms)", "Value": disk_access_answer}
    ]

    answer_table = cls.create_answer_table(
      headers=["Variable", "Value"],
      data_rows=answer_rows,
      answer_columns=["Value"]
    )

    # Use mixin to create complete body with both tables
    intro_text = "Given the information below, please calculate the following values."

    instructions = (
      "All calculated answers should be entered in milliseconds (ms). "
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
        "To calculate the total disk access time (or \"delay\"), we should first calculate each of the individual parts.",
        "Since we know that:",
      ])
    )
    explanation.add_element(
      ca.Equation(
        r"t_{total} = (\text{# of reads}) \cdot t_{access} + t_{transfer}"
      )
    )
    explanation.add_element(
      ca.Paragraph([
        "We therefore need to calculate ",
        ca.Equation(r"t_{access}", inline=True),
        " and ",
        ca.Equation(r"t_{transfer}", inline=True),
        ", where ",
        ca.Equation(r"t_{access} = t_{rotation} + t_{seek}", inline=True),
        ".",
      ])
    )
    
    explanation.add_elements([
      ca.Paragraph(["Starting with the rotation delay, we calculate:"]),
      ca.Equation(
        "t_{rotation} = "
        + f"\\frac{{1 \\text{{ minute}}}}{{{context['hard_drive_rotation_speed']} \\text{{ revolutions}}}}"
        + r"\cdot \frac{60 \text{ seconds}}{1 \text{ minute}} \cdot \frac{1000 \text{ ms}}{1 \text{ second}} \cdot \frac{1 \text{ revolution}}{2} = "
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
      ca.Paragraph([
        "Next we need to calculate our transfer delay, ",
        ca.Equation(r"t_{transfer}", inline=True),
        ", which we do as:",
      ]),
      ca.Equation(
        f"t_{{transfer}} "
        f"= \\frac{{{context['number_of_reads']} \\cdot {context['size_of_reads']} \\text{{ KB}}}}{{1}} \\cdot \\frac{{1 \\text{{ MB}}}}{{1024 \\text{{ KB}}}} "
        f"\\cdot \\frac{{1 \\text{{ second}}}}{{{context['transfer_rate']} \\text{{ MB}}}} \\cdot \\frac{{1000 \\text{{ ms}}}}{{1 \\text{{ second}}}} "
        f"= {context['transfer_delay']:0.2f}ms"
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
        "If we are given an inode number, there are a few steps that we need to take to load the actual inode.",
        "These consist of determining the address of the inode, which block would contain it, and then its address within the block.",
        "To find the inode address, we calculate:",
      ])
    )
    
    explanation.add_element(
      ca.Equation(
        r"(\text{Inode address}) = (\text{Inode Start Location}) + (\text{inode #}) \cdot (\text{inode size})"
      )
    )
    explanation.add_element(
      ca.Equation(
        f"= {context['inode_start_location']} + {context['inode_number']} \\cdot {context['inode_size']}"
      )
    )
    explanation.add_element(
      ca.Equation(
        f"= {context['inode_address']}"
      )
    )
    
    explanation.add_element(
      ca.Paragraph([
        "Next, we use this to figure out what block the inode is in. "
        "We do this directly so we know what block to load, "
        "thus minimizing the number of loads we have to make.",
      ])
    )
    explanation.add_element(ca.Equation(
      r"\text{Block containing inode} = (\text{Inode address}) \text{//} (\text{block size})"
    ))
    explanation.add_element(ca.Equation(
      f"= {context['inode_address']} \\text{{//}} {context['block_size']}"
    ))
    explanation.add_element(ca.Equation(
      f"= {context['inode_block']}"
    ))
    
    explanation.add_element(
      ca.Paragraph([
        "When we load this block, we now have in our system memory "
        "(remember, blocks on the hard drive are effectively useless to us until they're in main memory!), "
        "the inode, so next we need to figure out where it is within that block.",
        "This means that we'll need to find the offset into this block.",
        "We'll calculate this both as the offset in bytes and also in number of inodes, since we can use array indexing.",
      ])
    )
    
    explanation.add_element(ca.Equation(
      r"\text{offset within block} = (\text{Inode address}) \text{ mod } (\text{block size})"
    ))
    explanation.add_element(ca.Equation(
      f"= {context['inode_address']} \\text{{ mod }} {context['block_size']}"
    ))
    explanation.add_element(ca.Equation(
      f"= {context['inode_address_in_block']}"
    ))
    
    explanation.add_element(
      ca.Text("Remember that `mod` is the same as `%`, the modulo operation.")
    )
      
    explanation.add_element(ca.Equation(
      r"\text{index within block} = \frac{\text{offset within block}}{\text{inode size}}"
    ))
    explanation.add_element(ca.Equation(
      f"= \\frac{{{context['inode_address_in_block']}}}{{{context['inode_size']}}}"
    ))
    explanation.add_element(ca.Equation(
      f"= {context['inode_index_in_block']}"
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
    correct_cmd = str(operations[-1]["cmd"])
    baffles = [
      cmd
      for cmd in dict.fromkeys(op["cmd"] for op in operations[:-1])
      if cmd != correct_cmd
    ]

    command_answer = ca.Answer.dropdown(
      correct_cmd,
      baffles=baffles,
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
    
    lines_that_changed = []
    for start_line, end_line in zip(context["start_state"].split('\n'), context["end_state"].split('\n')):
      if start_line == end_line:
        continue
      lines_that_changed.append((start_line, end_line))
    
    def split_label_body(line: str) -> tuple[str, str]:
      match = re.match(r"^(?P<label>.+?)\s{2,}(?P<body>.+)$", line.rstrip())
      if match:
        return match.group("label"), match.group("body")
      parts = line.strip().split(None, 1)
      if len(parts) == 2:
        return parts[0], parts[1]
      return line.strip(), ""

    for start_line, end_line in lines_that_changed:
      label, start_body = split_label_body(start_line)
      _, end_body = split_label_body(end_line)
      explanation.add_element(ca.Paragraph([f"{label.title()}:"]))
      explanation.add_element(ca.Code(
        f"{start_body}\n{end_body}"
      ))
      explanation.add_element(ca.LineBreak())

    inode_bitmap_lines = list(filter(lambda s: "inode bitmap" in s[0], lines_that_changed))
    data_bitmap_lines = list(filter(lambda s: "data bitmap" in s[0], lines_that_changed))

    summary_lines = [
      "A great place to start is to check the bitmap changes, since they quickly tell us what kind of operation happened."
    ]

    if len(inode_bitmap_lines) > 0:
      inode_before, inode_after = inode_bitmap_lines[0]
      if inode_before.split()[-1].count("1") < inode_after.split()[-1].count("1"):
        summary_lines.append("The inode bitmap gained an inode, so we likely called `creat` or `mkdir`.")
      else:
        summary_lines.append("The inode bitmap lost an inode, so we likely called `unlink`.")

    if len(data_bitmap_lines) > 0:
      data_before, data_after = data_bitmap_lines[0]
      if data_before.split()[-1].count("1") < data_after.split()[-1].count("1"):
        summary_lines.append("The data bitmap gained a block, so we likely called `mkdir` or `write`.")
      else:
        summary_lines.append("The data bitmap lost a block, so we likely called `unlink`.")

    if len(data_bitmap_lines) == 0 and len(inode_bitmap_lines) == 0:
      summary_lines.append("If neither bitmap changed, then we likely called `link` or `unlink` and need to inspect the reference counts.")

    explanation.add_element(ca.Paragraph(summary_lines))

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

    explanation.add_element(ca.Paragraph(["The overall changes are highlighted with `*` symbols below"]))
    explanation.add_element(ca.Code(highlight_changes(context["start_state"], context["end_state"])))

    return explanation
