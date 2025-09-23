#!env python
from __future__ import annotations

import abc
import logging

from QuizGenerator.question import Question, Answer, QuestionRegistry
from QuizGenerator.contentast import ContentAST
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
      "answer__rotational_delay": Answer.float_value(
        "answer__rotational_delay",
        self.rotational_delay
      ),
      "answer__access_delay": Answer.float_value(
        "answer__access_delay",
        self.access_delay
      ),
      "answer__transfer_delay": Answer.float_value(
        "answer__transfer_delay",
        self.transfer_delay
      ),
      "answer__disk_access_delay": Answer.float_value(
        "answer__disk_access_delay",
        self.disk_access_delay
      ),
    })
  
  def get_body(self, *args, **kwargs) -> ContentAST.Section:
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
      f"Make sure your answers are rounded to {Answer.DEFAULT_ROUNDING_DIGITS} decimal points "
      f"(even if they are whole numbers), and do so after you finish all your calculations! "
      f"(i.e. don't use your rounded answers to calculate your overall answer)"
    )

    body = self.create_parameter_calculation_body(
      intro_text=intro_text,
      parameter_table=parameter_table,
      answer_table=answer_table,
      additional_instructions=instructions
    )

    return body
  
  def get_explanation(self) -> ContentAST.Section:
    explanation = ContentAST.Section()
    
    explanation.add_element(
      ContentAST.Paragraph([
        "To calculate the total disk access time (or \"delay\"), "
        "we should first calculate each of the individual parts.",
        r"Since we know that  $t_{total} = (\text{# of reads}) \cdot t_{access} + t_{transfer}$"
        r"we therefore need to calculate $t_{access}$ and  $t_{transfer}$, where "
        r"$t_{access} = t_{rotation} + t_{seek}$.",
      ])
    )
    
    explanation.add_elements([
      ContentAST.Paragraph(["Starting with the rotation delay, we calculate:"]),
      ContentAST.Equation(
        "t_{rotation} = "
        + f"\\frac{{1 minute}}{{{self.hard_drive_rotation_speed}revolutions}}"
        + r"\cdot \frac{60 seconds}{1 minute} \cdot \frac{1000 ms}{1 second} \cdot \frac{1 revolution}{2} = "
        + f"{self.rotational_delay:0.2f}ms",
      )
    ])
    
    explanation.add_elements([
      ContentAST.Paragraph([
        "Now we can calculate:",
      ]),
      ContentAST.Equation(
        f"t_{{access}} "
        f"= t_{{rotation}} + t_{{seek}} "
        f"= {self.rotational_delay:0.2f}ms + {self.seek_delay:0.2f}ms = {self.access_delay:0.2f}ms"
      )
    ])
    
    explanation.add_elements([
      ContentAST.Paragraph([r"Next we need to calculate our transfer delay, $t_{transfer}$, which we do as:"]),
      ContentAST.Equation(
        f"t_{{transfer}} "
        f"= \\frac{{{self.number_of_reads} \\cdot {self.size_of_reads}KB}}{{1}} \\cdot \\frac{{1MB}}{{1024KB}} "
        f"\\cdot \\frac{{1 second}}{{{self.transfer_rate}MB}} \\cdot \\frac{{1000ms}}{{1second}} "
        f"= {self.transfer_delay:0.2}ms"
      )
    ])
    
    explanation.add_elements([
      ContentAST.Paragraph(["Putting these together we get:"]),
      ContentAST.Equation(
        f"t_{{total}} "
        f"= \\text{{(# reads)}} \\cdot t_{{access}} + t_{{transfer}} "
        f"= {self.number_of_reads} \\cdot {self.access_delay:0.2f} + {self.transfer_delay:0.2f} "
        f"= {self.disk_access_delay:0.2f}ms")
    ])
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
      "answer__inode_address": Answer.integer("answer__inode_address", self.inode_address),
      "answer__inode_block": Answer.integer("answer__inode_block", self.inode_block),
      "answer__inode_address_in_block": Answer.integer("answer__inode_address_in_block", self.inode_address_in_block),
      "answer__inode_index_in_block": Answer.integer("answer__inode_index_in_block", self.inode_index_in_block),
    })
  
  def get_body(self) -> ContentAST.Section:
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

    instructions = (
      "(hint: they should all be round numbers). "
      "Remember, demonstrating you know the equations and what goes into them is generally sufficient."
    )

    body = self.create_parameter_calculation_body(
      intro_text=intro_text,
      parameter_table=parameter_table,
      answer_table=answer_table,
      additional_instructions=instructions
    )

    return body
  
  def get_explanation(self) -> ContentAST.Section:
    explanation = ContentAST.Section()
    
    explanation.add_element(
      ContentAST.Paragraph([
        "If we are given an inode number, there are a few steps that we need to take to load the actual inode.  "
        "These consist of determining the address of the inode, which block would contain it, "
        "and then its address within the block.",
        "To find the inode address, we calculate:",
      ])
    )
    
    explanation.add_element(
      ContentAST.Equation.make_block_equation__multiline_equals(
        r"(\text{Inode address})",
        [
          r"(\text{Inode Start Location}) + (\text{inode #}) \cdot (\text{inode size})",
          f"{self.inode_start_location} + {self.inode_number} \\cdot {self.inode_size}",
          f"{self.inode_address}"
        ])
    )
    
    explanation.add_element(
      ContentAST.Paragraph([
        "Next, we us this to figure out what block the inode is in.  "
        "We do this directly so we know what block to load, "
        "thus minimizing the number of loads we have to make.",
      ])
    )
    explanation.add_element(ContentAST.Equation.make_block_equation__multiline_equals(
      r"\text{Block containing inode}",
      [
        r"(\text{Inode address}) \mathbin{//} (\text{block size})",
        f"{self.inode_address} \\mathbin{{//}} {self.block_size}",
        f"{self.inode_block}"
      ]
    ))
    
    explanation.add_element(
      ContentAST.Paragraph([
        "When we load this block, we now have in our system memory "
        "(remember, blocks on the hard drive are effectively useless to us until they're in main memory!), "
        "the inode, so next we need to figure out where it is within that block."
        "This means that we'll need to find the offset into this block.  "
        "We'll calculate this both as the offset in bytes, and also in number of inodes, "
        "since we can use array indexing.",
      ])
    )
    
    explanation.add_element(ContentAST.Equation.make_block_equation__multiline_equals(
      r"\text{offset within block}",
      [
        r"(\text{Inode address}) \bmod (\text{block size})",
        f"{self.inode_address} \\bmod {self.block_size}",
        f"{self.inode_address_in_block}"
      ]
    ))
    
    explanation.add_element(
      ContentAST.Text("Remember that `mod` is the same as `%`, the modulo operation.")
    )
    
    explanation.add_element(ContentAST.Paragraph(["and"]))
      
    explanation.add_element(ContentAST.Equation.make_block_equation__multiline_equals(
      r"\text{index within block}",
      [
        r"\dfrac{\text{offset within block}}{\text{inode size}}",
        f"\\dfrac{{{self.inode_address_in_block}}}{{{self.inode_size}}}",
        f"{self.inode_index_in_block}"
      ]
    ))
    
    return explanation


@QuestionRegistry.register()
class VSFS_states(IOQuestion):

  from .ostep13_vsfs import fs as vsfs
  
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.answer_kind = Answer.AnswerKind.MULTIPLE_DROPDOWN
    
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
    
    self.answers["answer__cmd"] = Answer(
      "answer__cmd",
      f"{operations[-1]['cmd']}",
      kind=Answer.AnswerKind.MULTIPLE_DROPDOWN,
      correct=True,
      baffles=list(set([op['cmd'] for op in operations[:-1] if op != operations[-1]['cmd']]))
    )
  
  def get_body(self) -> ContentAST.Section:
    body = ContentAST.Section()
    
    body.add_element(ContentAST.Paragraph(["What operation happens between these two states?"]))
    
    body.add_element(
      ContentAST.Code(
        self.start_state,
        make_small=True
      )
    )
    
    body.add_element(
      ContentAST.AnswerBlock(
        ContentAST.Answer(
          self.answers["answer__cmd"],
          label="Command"
        )
      )
    )
    
    body.add_element(
      ContentAST.Code(
        self.end_state,
        make_small=True
      )
    )
    
    return body
  
  def get_explanation(self) -> ContentAST.Section:
    explanation = ContentAST.Section()
    
    explanation.add_element(
      ContentAST.Paragraph([
        "These questions are based on the VSFS simulator that our book mentions.  "
        "We will be discussing the interpretation of this in class, but you can also find information "
        "<a href=\"https://github.com/chyyuu/os_tutorial_lab/blob/master/ostep/ostep13-vsfs.md\">here</a>, "
        "as well as simulator code.  Please note that the code uses python 2.",
        "",
        "In general, I recommend looking for differences between the two outputs.  Recommended steps would be:",
        "<ol>"
        
        "<li> Check to see if there are differences between the bitmaps "
        "that could indicate a file/directroy were created or removed.</li>",
        
        "<li>Check the listed inodes to see if any entries have changed.  "
        "This might be a new entry entirely or a reference count changing.  "
        "If the references increased then this was likely a link or creation, "
        "and if it decreased then it is likely an unlink.</li>",
        
        "<li>Look at the data blocks to see if a new entry has "
        "been added to a directory or a new block has been mapped.</li>",
        
        "</ol>",
        "These steps can usually help you quickly identify "
        "what has occured in the simulation and key you in to the right answer."
      ])
    )
    
    return explanation
  