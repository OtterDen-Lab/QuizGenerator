#!env python
from __future__ import annotations

import abc
import collections
import copy
import enum
import logging
import math
from typing import List, Optional

from QuizGenerator.contentast import ContentAST
from QuizGenerator.question import Question, Answer, QuestionRegistry, RegenerableChoiceMixin
from QuizGenerator.mixins import TableQuestionMixin, BodyTemplatesMixin

log = logging.getLogger(__name__)


class MemoryQuestion(Question, abc.ABC):
  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.MEMORY)
    super().__init__(*args, **kwargs)


@QuestionRegistry.register("VirtualAddressParts")
class VirtualAddressParts(MemoryQuestion, TableQuestionMixin):
  MAX_BITS = 64
  
  class Target(enum.Enum):
    VA_BITS = "# VA Bits"
    VPN_BITS = "# VPN Bits"
    OFFSET_BITS = "# Offset Bits"
  
  def refresh(self, rng_seed=None, *args, **kwargs):
    super().refresh(rng_seed=rng_seed, *args, **kwargs)
    
    # Generate baselines, if not given
    self.num_bits_va = kwargs.get("num_bits_va", self.rng.randint(2, self.MAX_BITS))
    self.num_bits_offset = self.rng.randint(1, self.num_bits_va - 1)
    self.num_bits_vpn = self.num_bits_va - self.num_bits_offset
    
    self.possible_answers = {
      self.Target.VA_BITS: Answer.integer("answer__num_bits_va", self.num_bits_va),
      self.Target.OFFSET_BITS: Answer.integer("answer__num_bits_offset", self.num_bits_offset),
      self.Target.VPN_BITS: Answer.integer("answer__num_bits_vpn", self.num_bits_vpn)
    }
    
    # Select what kind of question we are going to be
    self.blank_kind = self.rng.choice(list(self.Target))
    
    self.answers['answer'] = self.possible_answers[self.blank_kind]
    
    return
  
  def get_body(self, **kwargs) -> ContentAST.Section:
    # Create table data with one blank cell
    table_data = [{}]
    for target in list(self.Target):
      if target == self.blank_kind:
        # This cell should be an answer blank
        table_data[0][target.value] = ContentAST.Answer(self.possible_answers[target], " bits")
      else:
        # This cell shows the value
        table_data[0][target.value] = f"{self.possible_answers[target].display} bits"
    
    table = self.create_fill_in_table(
      headers=[t.value for t in list(self.Target)],
      template_rows=table_data
    )
    
    body = ContentAST.Section()
    body.add_element(
      ContentAST.Paragraph(
        [
          "Given the information in the below table, please complete the table as appropriate."
        ]
      )
    )
    body.add_element(table)
    return body
  
  def get_explanation(self, **kwargs) -> ContentAST.Section:
    explanation = ContentAST.Section()
    
    explanation.add_element(
      ContentAST.Paragraph(
        [
          "Remember, when we are calculating the size of virtual address spaces, "
          "the number of bits in the overall address space is equal to the number of bits in the VPN "
          "plus the number of bits for the offset.",
          "We don't waste any bits!"
        ]
      )
    )
    
    explanation.add_element(
      ContentAST.Paragraph(
        [
          ContentAST.Text(f"{self.num_bits_va}", emphasis=(self.blank_kind == self.Target.VA_BITS)),
          ContentAST.Text(" = "),
          ContentAST.Text(f"{self.num_bits_vpn}", emphasis=(self.blank_kind == self.Target.VPN_BITS)),
          ContentAST.Text(" + "),
          ContentAST.Text(f"{self.num_bits_offset}", emphasis=(self.blank_kind == self.Target.OFFSET_BITS))
        ]
      )
    )
    
    return explanation


@QuestionRegistry.register()
class CachingQuestion(MemoryQuestion, RegenerableChoiceMixin, TableQuestionMixin, BodyTemplatesMixin):
  class Kind(enum.Enum):
    FIFO = enum.auto()
    LRU = enum.auto()
    BELADY = enum.auto()
    
    def __str__(self):
      return self.name
  
  class Cache:
    def __init__(self, kind: CachingQuestion.Kind, cache_size: int, all_requests: List[int] = None):
      self.kind = kind
      self.cache_size = cache_size
      self.all_requests = all_requests
      
      self.cache_state = []
      self.last_used = collections.defaultdict(lambda: -math.inf)
      self.frequency = collections.defaultdict(lambda: 0)
    
    def query_cache(self, request, request_number):
      was_hit = request in self.cache_state
      
      evicted = None
      if was_hit:
        # hit!
        pass
      else:
        # miss!
        if len(self.cache_state) == self.cache_size:
          # Then we are full and need to evict
          evicted = self.cache_state[0]
          self.cache_state = self.cache_state[1:]
        
        # Add to cache
        self.cache_state.append(request)
      
      # update state variable
      self.last_used[request] = request_number
      self.frequency[request] += 1
      
      # update cache state
      if self.kind == CachingQuestion.Kind.FIFO:
        pass
      elif self.kind == CachingQuestion.Kind.LRU:
        self.cache_state = sorted(
          self.cache_state,
          key=(lambda e: self.last_used[e]),
          reverse=False
        )
      # elif self.kind == CachingQuestion.Kind.LFU:
      #   self.cache_state = sorted(
      #     self.cache_state,
      #     key=(lambda e: (self.frequency[e], e)),
      #     reverse=False
      #   )
      elif self.kind == CachingQuestion.Kind.BELADY:
        upcoming_requests = self.all_requests[request_number + 1:]
        self.cache_state = sorted(
          self.cache_state,
          # key=(lambda e: (upcoming_requests.index(e), e) if e in upcoming_requests else (-math.inf, e)),
          key=(lambda e: (upcoming_requests.index(e), -e) if e in upcoming_requests else (math.inf, -e)),
          reverse=True
        )
      
      return (was_hit, evicted, self.cache_state)
  
  def __init__(self, *args, **kwargs):
    # Store parameters in kwargs for config_params BEFORE calling super().__init__()
    kwargs['num_elements'] = kwargs.get("num_elements", 5)
    kwargs['cache_size'] = kwargs.get("cache_size", 3)
    kwargs['num_requests'] = kwargs.get("num_requests", 10)

    # Register the regenerable choice using the mixin
    policy_str = (kwargs.get("policy") or kwargs.get("algo"))
    self.register_choice('policy', self.Kind, policy_str, kwargs)

    super().__init__(*args, **kwargs)

    self.num_elements = self.config_params.get("num_elements", 5)
    self.cache_size = self.config_params.get("cache_size", 3)
    self.num_requests = self.config_params.get("num_requests", 10)

  def refresh(self, previous: Optional[CachingQuestion] = None, *args, hard_refresh: bool = False, **kwargs):
    # Call parent refresh which seeds RNG and calls is_interesting()
    # Note: We ignore the parent's return value since we need to generate the workload first
    super().refresh(*args, **kwargs)

    # DEBUG: Log RNG state before get_choice
    rng_state_before = self.rng.getstate()
    log.debug(f"CachingQuestion.refresh: RNG state before get_choice: {rng_state_before[1][0]}")

    # Use the mixin to get the cache policy (randomly selected or fixed)
    self.cache_policy = self.get_choice('policy', self.Kind)

    # DEBUG: Log RNG state after get_choice
    rng_state_after = self.rng.getstate()
    log.debug(f"CachingQuestion.refresh: RNG state after get_choice (policy={self.cache_policy}): {rng_state_after[1][0]}")

    self.requests = (
        list(range(self.cache_size))  # Prime the cache with the compulsory misses
        + self.rng.choices(
      population=list(range(self.cache_size - 1)), k=1
    )  # Add in one request to an earlier  that will differentiate clearly between FIFO and LRU
        + self.rng.choices(
      population=list(range(self.cache_size, self.num_elements)), k=1
    )  ## Add in the rest of the requests
        + self.rng.choices(population=list(range(self.num_elements)), k=(self.num_requests - 2))
    ## Add in the rest of the requests
    )

    # DEBUG: Log the generated requests
    log.debug(f"CachingQuestion.refresh: Generated requests: {self.requests}")
    
    self.cache = CachingQuestion.Cache(self.cache_policy, self.cache_size, self.requests)
    
    self.request_results = {}
    number_of_hits = 0
    for (request_number, request) in enumerate(self.requests):
      was_hit, evicted, cache_state = self.cache.query_cache(request, request_number)
      if was_hit:
        number_of_hits += 1
      self.request_results[request_number] = {
        "request": (f"[answer__request]", request),
        "hit": (f"[answer__hit-{request_number}]", ('hit' if was_hit else 'miss')),
        "evicted": (f"[answer__evicted-{request_number}]", ('-' if evicted is None else f"{evicted}")),
        "cache_state": (f"[answer__cache_state-{request_number}]", ','.join(map(str, cache_state)))
      }
      
      self.answers.update(
        {
          f"answer__hit-{request_number}": Answer.string(
            f"answer__hit-{request_number}", ('hit' if was_hit else 'miss')
          ),
          f"answer__evicted-{request_number}": Answer.string(
            f"answer__evicted-{request_number}", ('-' if evicted is None else f"{evicted}")
          ),
          f"answer__cache_state-{request_number}": Answer.list_value(
            f"answer__cache_state-{request_number}", copy.copy(cache_state)
          ),
        }
      )
    
    self.hit_rate = 100 * number_of_hits / (self.num_requests)
    self.answers.update(
      {
        "answer__hit_rate": Answer.auto_float("answer__hit_rate", self.hit_rate)
      }
    )

    # Return whether this workload is interesting
    return self.is_interesting()
  
  def get_body(self, **kwargs) -> ContentAST.Section:
    # Create table data for cache simulation
    table_rows = []
    for request_number in sorted(self.request_results.keys()):
      table_rows.append(
        {
          "Page Requested": f"{self.requests[request_number]}",
          "Hit/Miss": f"answer__hit-{request_number}",  # Answer key
          "Evicted": f"answer__evicted-{request_number}",  # Answer key
          "Cache State": f"answer__cache_state-{request_number}"  # Answer key
        }
      )
    
    # Create table using mixin - automatically handles answer conversion
    cache_table = self.create_answer_table(
      headers=["Page Requested", "Hit/Miss", "Evicted", "Cache State"],
      data_rows=table_rows,
      answer_columns=["Hit/Miss", "Evicted", "Cache State"]
    )
    
    # Create hit rate answer block
    hit_rate_block = ContentAST.AnswerBlock(
      ContentAST.Answer(
        answer=self.answers["answer__hit_rate"],
        label=f"Hit rate, excluding compulsory misses.  If appropriate, round to {Answer.DEFAULT_ROUNDING_DIGITS} decimal digits.",
        unit="%"
      )
    )
    
    # Use mixin to create complete body
    intro_text = (
      f"Assume we are using a <b>{self.cache_policy}</b> caching policy and a cache size of <b>{self.cache_size}</b>. "
      "Given the below series of requests please fill in the table. "
      "For the hit/miss column, please write either \"hit\" or \"miss\". "
      "For the eviction column, please write either the number of the evicted page or simply a dash (e.g. \"-\")."
    )
    
    instructions = ContentAST.OnlyHtml([
      "For the cache state, please enter the cache contents in the order suggested in class, "
      "which means separated by commas with spaces (e.g. \"1, 2, 3\") "
      "and with the left-most being the next to be evicted. "
      "In the case where there is a tie, order by increasing number."
    ])
    
    body = self.create_fill_in_table_body(intro_text, instructions, cache_table)
    body.add_element(hit_rate_block)
    return body
  
  def get_explanation(self, **kwargs) -> ContentAST.Section:
    explanation = ContentAST.Section()
    
    explanation.add_element(ContentAST.Paragraph(["The full caching table can be seen below."]))
    
    explanation.add_element(
      ContentAST.Table(
        headers=["Page", "Hit/Miss", "Evicted", "Cache State"],
        data=[
          [
            self.request_results[request]["request"][1],
            self.request_results[request]["hit"][1],
            f'{self.request_results[request]["evicted"][1]}',
            f'{self.request_results[request]["cache_state"][1]}',
          ]
          for (request_number, request) in enumerate(sorted(self.request_results.keys()))
        ]
      )
    )
    
    explanation.add_element(
      ContentAST.Paragraph(
        [
          "To calculate the hit rate we calculate the percentage of requests "
          "that were cache hits out of the total number of requests. "
          f"In this case we are counting only all but {self.cache_size} requests, "
          f"since we are excluding compulsory misses."
        ]
      )
    )
    
    return explanation
  
  def is_interesting(self) -> bool:
    # todo: interesting is more likely based on whether I can differentiate between it and another algo,
    #  so maybe rerun with a different approach but same requests?
    return (self.hit_rate / 100.0) < 0.7


class MemoryAccessQuestion(MemoryQuestion, abc.ABC):
  PROBABILITY_OF_VALID = .875


@QuestionRegistry.register()
class BaseAndBounds(MemoryAccessQuestion, TableQuestionMixin, BodyTemplatesMixin):
  MAX_BITS = 32
  MIN_BOUNDS_BIT = 5
  MAX_BOUNDS_BITS = 16
  
  def refresh(self, rng_seed=None, *args, **kwargs):
    super().refresh(rng_seed=rng_seed, *args, **kwargs)
    
    max_bound_bits = kwargs.get("max_bound_bits")
    
    bounds_bits = self.rng.randint(
      self.MIN_BOUNDS_BIT,
      self.MAX_BOUNDS_BITS
    )
    base_bits = self.MAX_BITS - bounds_bits
    
    self.bounds = int(math.pow(2, bounds_bits))
    self.base = self.rng.randint(1, int(math.pow(2, base_bits))) * self.bounds
    self.virtual_address = self.rng.randint(1, int(self.bounds / self.PROBABILITY_OF_VALID))
    
    if self.virtual_address < self.bounds:
      self.answers["answer"] = Answer.binary_hex(
        "answer__physical_address",
        self.base + self.virtual_address,
        length=math.ceil(math.log2(self.base + self.virtual_address))
      )
    else:
      self.answers["answer"] = Answer.string("answer__physical_address", "INVALID")
  
  def get_body(self) -> ContentAST.Section:
    # Use mixin to create parameter table with answer
    parameter_info = {
      "Base": f"0x{self.base:X}",
      "Bounds": f"0x{self.bounds:X}",
      "Virtual Address": f"0x{self.virtual_address:X}"
    }
    
    table = self.create_parameter_answer_table(
      parameter_info=parameter_info,
      answer_label="Physical Address",
      answer_key="answer",
      transpose=True
    )
    
    return self.create_parameter_calculation_body(
      intro_text=(
        "Given the information in the below table, "
        "please calcuate the physical address associated with the given virtual address. "
        "If the virtual address is invalid please simply write ***INVALID***."
      ),
      parameter_table=table
    )
  
  def get_explanation(self) -> ContentAST.Section:
    explanation = ContentAST.Section()
    
    explanation.add_element(
      ContentAST.Paragraph(
        [
          "There's two steps to figuring out base and bounds.",
          "1. Are we within the bounds?\n",
          "2. If so, add to our base.\n",
          "",
        ]
      )
    )
    
    explanation.add_element(
      ContentAST.Paragraph(
        [
          f"Step 1: 0x{self.virtual_address:X} < 0x{self.bounds:X} "
          f"--> {'***VALID***' if (self.virtual_address < self.bounds) else 'INVALID'}"
        ]
      )
    )
    
    if self.virtual_address < self.bounds:
      explanation.add_element(
        ContentAST.Paragraph(
          [
            f"Step 2: Since the previous check passed, we calculate "
            f"0x{self.base:X} + 0x{self.virtual_address:X} "
            f"= ***0x{self.base + self.virtual_address:X}***.",
            "If it had been invalid we would have simply written INVALID"
          ]
        )
      )
    else:
      explanation.add_element(
        ContentAST.Paragraph(
          [
            f"Step 2: Since the previous check failed, we simply write ***INVALID***.",
            "***If*** it had been valid, we would have calculated "
            f"0x{self.base:X} + 0x{self.virtual_address:X} "
            f"= 0x{self.base + self.virtual_address:X}.",
          ]
        )
      )
    
    return explanation


@QuestionRegistry.register()
class Segmentation(MemoryAccessQuestion, TableQuestionMixin, BodyTemplatesMixin):
  MAX_BITS = 20
  MIN_VIRTUAL_BITS = 5
  MAX_VIRTUAL_BITS = 10
  
  def __within_bounds(self, segment, offset, bounds):
    if segment == "unallocated":
      return False
    elif bounds < offset:
      return False
    else:
      return True
  
  def refresh(self, *args, **kwargs):
    super().refresh(*args, **kwargs)
    
    # Pick how big each of our address spaces will be
    self.virtual_bits = self.rng.randint(self.MIN_VIRTUAL_BITS, self.MAX_VIRTUAL_BITS)
    self.physical_bits = self.rng.randint(self.virtual_bits + 1, self.MAX_BITS)
    
    # Start with blank base and bounds
    self.base = {
      "code": 0,
      "heap": 0,
      "stack": 0,
    }
    self.bounds = {
      "code": 0,
      "heap": 0,
      "stack": 0,
    }
    
    min_bounds = 4
    max_bounds = int(2 ** (self.virtual_bits - 2))
    
    def segment_collision(base, bounds):
      # lol, I think this is probably silly, but should work
      return 0 != len(
        set.intersection(
          *[
            set(range(base[segment], base[segment] + bounds[segment] + 1))
            for segment in base.keys()
          ]
        )
      )
    
    self.base["unallocated"] = 0
    self.bounds["unallocated"] = 0
    
    # Make random placements and check to make sure they are not overlapping
    while (segment_collision(self.base, self.bounds)):
      for segment in self.base.keys():
        self.bounds[segment] = self.rng.randint(min_bounds, max_bounds - 1)
        self.base[segment] = self.rng.randint(0, (2 ** self.physical_bits - self.bounds[segment]))
    
    # Pick a random segment for us to use
    self.segment = self.rng.choice(list(self.base.keys()))
    self.segment_bits = {
      "code": 0,
      "heap": 1,
      "unallocated": 2,
      "stack": 3
    }[self.segment]
    
    # Try to pick a random address within that range
    try:
      self.offset = self.rng.randint(
        1,
        min(
          [
            max_bounds - 1,
            int(self.bounds[self.segment] / self.PROBABILITY_OF_VALID)
          ]
        )
      )
    except KeyError:
      # If we are in an unallocated section, we'll get a key error (I think)
      self.offset = self.rng.randint(0, max_bounds - 1)
    
    # Calculate a virtual address based on the segment and the offset
    self.virtual_address = (
        (self.segment_bits << (self.virtual_bits - 2))
        + self.offset
    )
    
    # Calculate physical address based on offset
    self.physical_address = self.base[self.segment] + self.offset
    
    # Set answers based on whether it's in bounds or not
    if self.__within_bounds(self.segment, self.offset, self.bounds[self.segment]):
      self.answers["answer__physical_address"] = Answer.binary_hex(
        "answer__physical_address",
        self.physical_address,
        length=self.physical_bits
      )
    else:
      self.answers["answer__physical_address"] = Answer.string(
        "answer__physical_address",
        "INVALID"
      )
    
    self.answers["answer__segment"] = Answer.string("answer__segment", self.segment)
  
  def get_body(self) -> ContentAST.Section:
    body = ContentAST.Section()
    
    body.add_element(
      ContentAST.Paragraph(
        [
          f"Given a virtual address space of {self.virtual_bits}bits, "
          f"and a physical address space of {self.physical_bits}bits, "
          "what is the physical address associated with the virtual address "
          f"0b{self.virtual_address:0{self.virtual_bits}b}?",
          "If it is invalid simply type INVALID.",
          "Note: assume that the stack grows in the same way as the code and the heap."
        ]
      )
    )
    
    # Create segment table using mixin
    segment_rows = [
      {"": "code", "base": f"0b{self.base['code']:0{self.physical_bits}b}", "bounds": f"0b{self.bounds['code']:0b}"},
      {"": "heap", "base": f"0b{self.base['heap']:0{self.physical_bits}b}", "bounds": f"0b{self.bounds['heap']:0b}"},
      {"": "stack", "base": f"0b{self.base['stack']:0{self.physical_bits}b}", "bounds": f"0b{self.bounds['stack']:0b}"}
    ]
    
    segment_table = self.create_answer_table(
      headers=["", "base", "bounds"],
      data_rows=segment_rows,
      answer_columns=[]  # No answer columns in this table
    )
    
    body.add_element(segment_table)
    
    body.add_element(
      ContentAST.AnswerBlock(
        [
          ContentAST.Answer(
            self.answers["answer__segment"],
            label="Segment name"
          ),
          ContentAST.Answer(
            self.answers["answer__physical_address"],
            label="Physical Address"
          )
        ]
      )
    )
    return body
  
  def get_explanation(self) -> ContentAST.Section:
    explanation = ContentAST.Section()
    
    explanation.add_element(
      ContentAST.Paragraph(
        [
          "The core idea to keep in mind with segmentation is that you should always check ",
          "the first two bits of the virtual address to see what segment it is in and then go from there."
          "Keep in mind, "
          "we also may need to include padding if our virtual address has a number of leading zeros left off!"
        ]
      )
    )
    
    explanation.add_element(
      ContentAST.Paragraph(
        [
          f"In this problem our virtual address, "
          f"converted to binary and including padding, is 0b{self.virtual_address:0{self.virtual_bits}b}.",
          f"From this we know that our segment bits are 0b{self.segment_bits:02b}, "
          f"meaning that we are in the ***{self.segment}*** segment.",
          ""
        ]
      )
    )
    
    if self.segment == "unallocated":
      explanation.add_element(
        ContentAST.Paragraph(
          [
            "Since this is the unallocated segment there are no possible valid translations, so we enter ***INVALID***."
          ]
        )
      )
    else:
      explanation.add_element(
        ContentAST.Paragraph(
          [
            f"Since we are in the {self.segment} segment, "
            f"we see from our table that our bounds are {self.bounds[self.segment]}. "
            f"Remember that our check for our {self.segment} segment is: ",
            f"`if (offset > bounds({self.segment})) : INVALID`",
            "which becomes"
            f"`if ({self.offset:0b} > {self.bounds[self.segment]:0b}) : INVALID`"
          ]
        )
      )
      
      if not self.__within_bounds(self.segment, self.offset, self.bounds[self.segment]):
        # then we are outside of bounds
        explanation.add_element(
          ContentAST.Paragraph(
            [
              "We can therefore see that we are outside of bounds so we should put ***INVALID***.",
              "If we <i>were</i> requesting a valid memory location we could use the below steps to do so."
              "<hr>"
            ]
          )
        )
      else:
        explanation.add_element(
          ContentAST.Paragraph(
            [
              "We are therefore in bounds so we can calculate our physical address, as we do below."
            ]
          )
        )
      
      explanation.add_element(
        ContentAST.Paragraph(
          [
            "To find the physical address we use the formula:",
            "<code>physical_address = base(segment) + offset</code>",
            "which becomes",
            f"<code>physical_address = {self.base[self.segment]:0b} + {self.offset:0b}</code>.",
            ""
          ]
        )
      )
      
      explanation.add_element(
        ContentAST.Paragraph(
          [
            "Lining this up for ease we can do this calculation as:"
          ]
        )
      )
      explanation.add_element(
        ContentAST.Code(
          f"  0b{self.base[self.segment]:0{self.physical_bits}b}\n"
          f"<u>+ 0b{self.offset:0{self.physical_bits}b}</u>\n"
          f"  0b{self.physical_address:0{self.physical_bits}b}\n"
        )
      )
    
    return explanation


@QuestionRegistry.register()
class Paging(MemoryAccessQuestion, TableQuestionMixin, BodyTemplatesMixin):
  MIN_OFFSET_BITS = 3
  MIN_VPN_BITS = 3
  MIN_PFN_BITS = 3
  
  MAX_OFFSET_BITS = 8
  MAX_VPN_BITS = 8
  MAX_PFN_BITS = 16
  
  def refresh(self, rng_seed=None, *args, **kwargs):
    super().refresh(rng_seed=rng_seed, *args, **kwargs)
    
    self.num_bits_offset = self.rng.randint(self.MIN_OFFSET_BITS, self.MAX_OFFSET_BITS)
    self.num_bits_vpn = self.rng.randint(self.MIN_VPN_BITS, self.MAX_VPN_BITS)
    self.num_bits_pfn = self.rng.randint(max([self.MIN_PFN_BITS, self.num_bits_vpn]), self.MAX_PFN_BITS)
    
    self.virtual_address = self.rng.randint(1, 2 ** (self.num_bits_vpn + self.num_bits_offset))
    
    # Calculate these two
    self.offset = self.virtual_address % (2 ** (self.num_bits_offset))
    self.vpn = self.virtual_address // (2 ** (self.num_bits_offset))
    
    # Generate this randomly
    self.pfn = self.rng.randint(0, 2 ** (self.num_bits_pfn))
    
    # Calculate this
    self.physical_address = self.pfn * (2 ** self.num_bits_offset) + self.offset
    
    if self.rng.choices([True, False], weights=[(self.PROBABILITY_OF_VALID), (1 - self.PROBABILITY_OF_VALID)], k=1)[0]:
      self.is_valid = True
      # Set our actual entry to be in the table and valid
      self.pte = self.pfn + (2 ** (self.num_bits_pfn))
      # self.physical_address_var = VariableHex("Physical Address", self.physical_address, num_bits=(self.num_pfn_bits+self.num_offset_bits), default_presentation=VariableHex.PRESENTATION.BINARY)
      # self.pfn_var = VariableHex("PFN", self.pfn, num_bits=self.num_pfn_bits, default_presentation=VariableHex.PRESENTATION.BINARY)
    else:
      self.is_valid = False
      # Leave it as invalid
      self.pte = self.pfn
      # self.physical_address_var = Variable("Physical Address", "INVALID")
      # self.pfn_var = Variable("PFN",  "INVALID")
    
    # self.pte_var = VariableHex("PTE", self.pte, num_bits=(self.num_pfn_bits+1), default_presentation=VariableHex.PRESENTATION.BINARY)
    
    self.answers.update(
      {
        "answer__vpn": Answer.binary_hex("answer__vpn", self.vpn, length=self.num_bits_vpn),
        "answer__offset": Answer.binary_hex("answer__offset", self.offset, length=self.num_bits_offset),
        "answer__pte": Answer.binary_hex("answer__pte", self.pte, length=(self.num_bits_pfn + 1)),
      }
    )
    
    if self.is_valid:
      self.answers.update(
        {
          "answer__is_valid": Answer.string("answer__is_valid", "VALID"),
          "answer__pfn": Answer.binary_hex("answer__pfn", self.pfn, length=self.num_bits_pfn),
          "answer__physical_address": Answer.binary_hex(
            "answer__physical_address", self.physical_address, length=(self.num_bits_pfn + self.num_bits_offset)
          ),
        }
      )
    else:
      self.answers.update(
        {
          "answer__is_valid": Answer.string("answer__is_valid", "INVALID"),
          "answer__pfn": Answer.string("answer__pfn", "INVALID"),
          "answer__physical_address": Answer.string("answer__physical_address", "INVALID"),
        }
      )
  
  def get_body(self, *args, **kwargs) -> ContentAST.Section:
    body = ContentAST.Section()
    
    body.add_element(
      ContentAST.Paragraph(
        [
          "Given the below information please calculate the equivalent physical address of the given virtual address, filling out all steps along the way.",
          "Remember, we typically have the MSB representing valid or invalid."
        ]
      )
    )
    
    # Create parameter info table using mixin
    parameter_info = {
      "Virtual Address": f"0b{self.virtual_address:0{self.num_bits_vpn + self.num_bits_offset}b}",
      "# VPN bits": f"{self.num_bits_vpn}",
      "# PFN bits": f"{self.num_bits_pfn}"
    }
    
    body.add_element(self.create_info_table(parameter_info))
    
    # Make values for Page Table
    table_size = self.rng.randint(5, 8)
    
    lowest_possible_bottom = max([0, self.vpn - table_size])
    highest_possible_bottom = min([2 ** self.num_bits_vpn - table_size, self.vpn])
    
    table_bottom = self.rng.randint(lowest_possible_bottom, highest_possible_bottom)
    table_top = table_bottom + table_size
    
    page_table = {}
    page_table[self.vpn] = self.pte
    
    # Fill in the rest of the table
    # for vpn in range(2**self.num_vpn_bits):
    for vpn in range(table_bottom, table_top):
      if vpn == self.vpn: continue
      pte = page_table[self.vpn]
      while pte in page_table.values():
        pte = self.rng.randint(0, 2 ** self.num_bits_pfn - 1)
        if self.rng.choices([True, False], weights=[(1 - self.PROBABILITY_OF_VALID), self.PROBABILITY_OF_VALID], k=1)[
          0]:
          # Randomly set it to be valid
          pte += (2 ** (self.num_bits_pfn))
      # Once we have a unique random entry, put it into the Page Table
      page_table[vpn] = pte
    
    # Add in ellipses before and after page table entries, if appropriate
    value_matrix = []
    
    if min(page_table.keys()) != 0:
      value_matrix.append(["...", "..."])
    
    value_matrix.extend(
      [
        [f"0b{vpn:0{self.num_bits_vpn}b}", f"0b{pte:0{(self.num_bits_pfn + 1)}b}"]
        for vpn, pte in sorted(page_table.items())
      ]
    )
    
    if (max(page_table.keys()) + 1) != 2 ** self.num_bits_vpn:
      value_matrix.append(["...", "..."])
    
    body.add_element(
      ContentAST.Table(
        headers=["VPN", "PTE"],
        data=value_matrix
      )
    )
    
    body.add_element(
      ContentAST.AnswerBlock(
        [
          
          ContentAST.Answer(self.answers["answer__vpn"], label="VPN"),
          ContentAST.Answer(self.answers["answer__offset"], label="Offset"),
          ContentAST.Answer(self.answers["answer__pte"], label="PTE"),
          ContentAST.Answer(self.answers["answer__is_valid"], label="VALID or INVALID?"),
          ContentAST.Answer(self.answers["answer__pfn"], label="PFN"),
          ContentAST.Answer(self.answers["answer__physical_address"], label="Physical Address"),
        ]
      )
    )
    
    return body
  
  def get_explanation(self, *args, **kwargs) -> ContentAST.Section:
    explanation = ContentAST.Section()
    
    explanation.add_element(
      ContentAST.Paragraph(
        [
          "The core idea of Paging is we want to break the virtual address into the VPN and the offset.  "
          "From here, we get the Page Table Entry corresponding to the VPN, and check the validity of the entry.  "
          "If it is valid, we clear the metadata and attach the PFN to the offset and have our physical address.",
        ]
      )
    )
    
    explanation.add_element(
      ContentAST.Paragraph(
        [
          "Don't forget to pad with the appropriate number of 0s (the appropriate number is the number of bits)!",
        ]
      )
    )
    
    explanation.add_element(
      ContentAST.Paragraph(
        [
          f"Virtual Address = VPN | offset",
          f"<tt>0b{self.virtual_address:0{self.num_bits_vpn + self.num_bits_offset}b}</tt> "
          f"= <tt>0b{self.vpn:0{self.num_bits_vpn}b}</tt> | <tt>0b{self.offset:0{self.num_bits_offset}b}</tt>",
        ]
      )
    )
    
    explanation.add_element(
      ContentAST.Paragraph(
        [
          "We next use our VPN to index into our page table and find the corresponding entry."
          f"Our Page Table Entry is ",
          f"<tt>0b{self.pte:0{(self.num_bits_pfn + 1)}b}</tt>"
          f"which we found by looking for our VPN in the page table.",
        ]
      )
    )
    
    if self.is_valid:
      explanation.add_element(
        ContentAST.Paragraph(
          [
            f"In our PTE we see that the first bit is <b>{self.pte // (2 ** self.num_bits_pfn)}</b> meaning that the translation is <b>VALID</b>"
          ]
        )
      )
    else:
      explanation.add_element(
        ContentAST.Paragraph(
          [
            f"In our PTE we see that the first bit is <b>{self.pte // (2 ** self.num_bits_pfn)}</b> meaning that the translation is <b>INVALID</b>.",
            "Therefore, we just write \"INVALID\" as our answer.",
            "If it were valid we would complete the below steps.",
            "<hr>"
          ]
        )
      )
    
    explanation.add_element(
      ContentAST.Paragraph(
        [
          "Next, we convert our PTE to our PFN by removing our metadata.  "
          "In this case we're just removing the leading bit.  We can do this by applying a binary mask.",
          f"PFN = PTE & mask",
          f"which is,"
        ]
      )
    )
    explanation.add_element(
      ContentAST.Equation(
        f"\\texttt{{{self.pfn:0{self.num_bits_pfn}b}}} "
        f"= \\texttt{{0b{self.pte:0{self.num_bits_pfn + 1}b}}} "
        f"\\& \\texttt{{0b{(2 ** self.num_bits_pfn) - 1:0{self.num_bits_pfn + 1}b}}}"
      )
    )
    
    explanation.add_elements(
      [
        ContentAST.Paragraph(
          [
            "We then add combine our PFN and offset, "
            "Physical Address = PFN | offset",
          ]
        ),
        ContentAST.Equation(
          fr"{r'\mathbf{' if self.is_valid else ''}\mathtt{{0b{self.physical_address:0{self.num_bits_pfn + self.num_bits_offset}b}}}{r'}' if self.is_valid else ''} = \mathtt{{0b{self.pfn:0{self.num_bits_pfn}b}}} \mid \mathtt{{0b{self.offset:0{self.num_bits_offset}b}}}"
        )
      ]
    )
    
    explanation.add_elements(
      [
        ContentAST.Paragraph(["Note: Strictly speaking, this calculation is:", ]),
        ContentAST.Equation(
          fr"{r'\mathbf{' if self.is_valid else ''}\mathtt{{0b{self.physical_address:0{self.num_bits_pfn + self.num_bits_offset}b}}}{r'}' if self.is_valid else ''} = \mathtt{{0b{self.pfn:0{self.num_bits_pfn}b}{0:0{self.num_bits_offset}}}} + \mathtt{{0b{self.offset:0{self.num_bits_offset}b}}}"
        ),
        ContentAST.Paragraph(["But that's a lot of extra 0s, so I'm splitting them up for succinctness"])
      ]
    )

    return explanation


@QuestionRegistry.register()
class HierarchicalPaging(MemoryAccessQuestion, TableQuestionMixin, BodyTemplatesMixin):
  MIN_OFFSET_BITS = 3
  MIN_PDI_BITS = 2
  MIN_PTI_BITS = 2
  MIN_PFN_BITS = 4

  MAX_OFFSET_BITS = 5
  MAX_PDI_BITS = 3
  MAX_PTI_BITS = 3
  MAX_PFN_BITS = 6

  def refresh(self, rng_seed=None, *args, **kwargs):
    super().refresh(rng_seed=rng_seed, *args, **kwargs)

    # Set up bit counts
    self.num_bits_offset = self.rng.randint(self.MIN_OFFSET_BITS, self.MAX_OFFSET_BITS)
    self.num_bits_pdi = self.rng.randint(self.MIN_PDI_BITS, self.MAX_PDI_BITS)
    self.num_bits_pti = self.rng.randint(self.MIN_PTI_BITS, self.MAX_PTI_BITS)
    self.num_bits_pfn = self.rng.randint(self.MIN_PFN_BITS, self.MAX_PFN_BITS)

    # Total VPN bits = PDI + PTI
    self.num_bits_vpn = self.num_bits_pdi + self.num_bits_pti
 
    # Generate a random virtual address
    self.virtual_address = self.rng.randint(1, 2 ** (self.num_bits_vpn + self.num_bits_offset))

    # Extract components from virtual address
    self.offset = self.virtual_address % (2 ** self.num_bits_offset)
    vpn = self.virtual_address // (2 ** self.num_bits_offset)

    self.pti = vpn % (2 ** self.num_bits_pti)
    self.pdi = vpn // (2 ** self.num_bits_pti)

    # Generate PFN randomly
    self.pfn = self.rng.randint(0, 2 ** self.num_bits_pfn - 1)

    # Calculate physical address
    self.physical_address = self.pfn * (2 ** self.num_bits_offset) + self.offset

    # Determine validity at both levels
    # PD entry can be valid or invalid
    self.pd_valid = self.rng.choices([True, False], weights=[self.PROBABILITY_OF_VALID, 1 - self.PROBABILITY_OF_VALID], k=1)[0]

    # PT entry only matters if PD is valid
    if self.pd_valid:
      self.pt_valid = self.rng.choices([True, False], weights=[self.PROBABILITY_OF_VALID, 1 - self.PROBABILITY_OF_VALID], k=1)[0]
    else:
      self.pt_valid = False  # Doesn't matter, won't be checked

    # Generate a page table number (PTBR - Page Table Base Register value in the PD entry)
    # This represents which page table to use
    self.page_table_number = self.rng.randint(0, 2 ** self.num_bits_pfn - 1)

    # Create PD entry: valid bit + page table number
    if self.pd_valid:
      self.pd_entry = (2 ** self.num_bits_pfn) + self.page_table_number
    else:
      self.pd_entry = self.page_table_number  # Invalid, no valid bit set

    # Create PT entry: valid bit + PFN
    if self.pt_valid:
      self.pte = (2 ** self.num_bits_pfn) + self.pfn
    else:
      self.pte = self.pfn  # Invalid, no valid bit set

    # Overall validity requires both levels to be valid
    self.is_valid = self.pd_valid and self.pt_valid

    # Build page directory - show 3-4 entries
    pd_size = self.rng.randint(3, 4)
    lowest_pd_bottom = max([0, self.pdi - pd_size])
    highest_pd_bottom = min([2 ** self.num_bits_pdi - pd_size, self.pdi])
    pd_bottom = self.rng.randint(lowest_pd_bottom, highest_pd_bottom)
    pd_top = pd_bottom + pd_size

    self.page_directory = {}
    self.page_directory[self.pdi] = self.pd_entry

    # Fill in other PD entries
    for pdi in range(pd_bottom, pd_top):
      if pdi == self.pdi:
        continue
      # Generate random PD entry
      pt_num = self.rng.randint(0, 2 ** self.num_bits_pfn - 1)
      while pt_num == self.page_table_number:  # Make sure it's different
        pt_num = self.rng.randint(0, 2 ** self.num_bits_pfn - 1)

      # Randomly valid or invalid
      if self.rng.choices([True, False], weights=[self.PROBABILITY_OF_VALID, 1 - self.PROBABILITY_OF_VALID], k=1)[0]:
        pd_val = (2 ** self.num_bits_pfn) + pt_num
      else:
        pd_val = pt_num

      self.page_directory[pdi] = pd_val

    # Build 2-3 page tables to show
    # Always include the one we need, plus 1-2 others
    num_page_tables_to_show = self.rng.randint(2, 3)

    # Get unique page table numbers from the PD entries (extract PT numbers from valid entries)
    shown_pt_numbers = set()
    for pdi, pd_val in self.page_directory.items():
      pt_num = pd_val % (2 ** self.num_bits_pfn)  # Extract PT number (remove valid bit)
      shown_pt_numbers.add(pt_num)

    # Ensure our required page table is included
    shown_pt_numbers.add(self.page_table_number)

    # Limit to requested number, but ALWAYS keep the required page table
    shown_pt_numbers_list = list(shown_pt_numbers)
    if self.page_table_number in shown_pt_numbers_list:
      # Remove it temporarily so we can add it back first
      shown_pt_numbers_list.remove(self.page_table_number)
    # Start with required page table, then add others up to the limit
    shown_pt_numbers = [self.page_table_number] + shown_pt_numbers_list[:num_page_tables_to_show - 1]

    # Build each page table
    self.page_tables = {}  # Dict mapping PT number -> dict of PTI -> PTE

    # Use consistent size for all page tables for cleaner presentation
    pt_size = self.rng.randint(2, 4)

    # Determine the PTI range that all tables will use (based on target PTI)
    # This ensures all tables show the same PTI values for consistency
    lowest_pt_bottom = max([0, self.pti - pt_size + 1])
    highest_pt_bottom = min([2 ** self.num_bits_pti - pt_size, self.pti])
    pt_bottom = self.rng.randint(lowest_pt_bottom, highest_pt_bottom)
    pt_top = pt_bottom + pt_size

    # Generate all page tables using the SAME PTI range
    for pt_num in shown_pt_numbers:
      self.page_tables[pt_num] = {}

      for pti in range(pt_bottom, pt_top):
        if pt_num == self.page_table_number and pti == self.pti:
          # Use the actual answer for the target page table entry
          self.page_tables[pt_num][pti] = self.pte
        else:
          # Generate random PTE for all other entries
          pfn = self.rng.randint(0, 2 ** self.num_bits_pfn - 1)
          if self.rng.choices([True, False], weights=[self.PROBABILITY_OF_VALID, 1 - self.PROBABILITY_OF_VALID], k=1)[0]:
            pte_val = (2 ** self.num_bits_pfn) + pfn
          else:
            pte_val = pfn

          self.page_tables[pt_num][pti] = pte_val

    # Set up answers
    self.answers.update({
      "answer__pdi": Answer.binary_hex("answer__pdi", self.pdi, length=self.num_bits_pdi),
      "answer__pti": Answer.binary_hex("answer__pti", self.pti, length=self.num_bits_pti),
      "answer__offset": Answer.binary_hex("answer__offset", self.offset, length=self.num_bits_offset),
      "answer__pd_entry": Answer.binary_hex("answer__pd_entry", self.pd_entry, length=(self.num_bits_pfn + 1)),
      "answer__pt_number": Answer.binary_hex("answer__pt_number", self.page_table_number, length=self.num_bits_pfn) if self.pd_valid else Answer.string("answer__pt_number", "INVALID"),
    })

    # PTE answer: if PD is valid, accept the actual PTE value from the table
    # (regardless of whether that PTE is valid or invalid)
    if self.pd_valid:
      self.answers.update({
        "answer__pte": Answer.binary_hex("answer__pte", self.pte, length=(self.num_bits_pfn + 1)),
      })
    else:
      # If PD is invalid, student can't look up the page table
      # Accept both "INVALID" (for consistency) and "N/A" (for accuracy)
      self.answers.update({
        "answer__pte": Answer.string("answer__pte", ["INVALID", "N/A"]),
      })

    # Validity, PFN, and Physical Address depend on BOTH levels being valid
    if self.pd_valid and self.pt_valid:
      self.answers.update({
        "answer__is_valid": Answer.string("answer__is_valid", "VALID"),
        "answer__pfn": Answer.binary_hex("answer__pfn", self.pfn, length=self.num_bits_pfn),
        "answer__physical_address": Answer.binary_hex(
          "answer__physical_address", self.physical_address, length=(self.num_bits_pfn + self.num_bits_offset)
        ),
      })
    else:
      self.answers.update({
        "answer__is_valid": Answer.string("answer__is_valid", "INVALID"),
        "answer__pfn": Answer.string("answer__pfn", "INVALID"),
        "answer__physical_address": Answer.string("answer__physical_address", "INVALID"),
      })

  def get_body(self, *args, **kwargs) -> ContentAST.Section:
    body = ContentAST.Section()

    body.add_element(
      ContentAST.Paragraph([
        "Given the below information please calculate the equivalent physical address of the given virtual address, filling out all steps along the way.",
        "This problem uses <b>two-level (hierarchical) paging</b>.",
        "Remember, we typically have the MSB representing valid or invalid."
      ])
    )

    # Create parameter info table using mixin (same format as Paging question)
    parameter_info = {
      "Virtual Address": f"0b{self.virtual_address:0{self.num_bits_vpn + self.num_bits_offset}b}",
      "# PDI bits": f"{self.num_bits_pdi}",
      "# PTI bits": f"{self.num_bits_pti}",
      "# Offset bits": f"{self.num_bits_offset}",
      "# PFN bits": f"{self.num_bits_pfn}"
    }

    body.add_element(self.create_info_table(parameter_info))

    # Page Directory table
    pd_matrix = []
    if min(self.page_directory.keys()) != 0:
      pd_matrix.append(["...", "..."])

    pd_matrix.extend([
      [f"0b{pdi:0{self.num_bits_pdi}b}", f"0b{pd_val:0{self.num_bits_pfn + 1}b}"]
      for pdi, pd_val in sorted(self.page_directory.items())
    ])

    if (max(self.page_directory.keys()) + 1) != 2 ** self.num_bits_pdi:
      pd_matrix.append(["...", "..."])

    # Use a simple text paragraph - the bold will come from markdown conversion
    body.add_element(
      ContentAST.Paragraph([
        "**Page Directory:**"
      ])
    )
    body.add_element(
      ContentAST.Table(
        headers=["PDI", "PD Entry"],
        data=pd_matrix
      )
    )

    # Page Tables - use TableGroup for side-by-side display
    table_group = ContentAST.TableGroup()

    for pt_num in sorted(self.page_tables.keys()):
      pt_matrix = []
      pt_entries = self.page_tables[pt_num]

      min_pti = min(pt_entries.keys())
      max_pti = max(pt_entries.keys())
      max_possible_pti = 2 ** self.num_bits_pti - 1

      # Smart leading ellipsis: only if there are 2+ hidden entries before
      # (if only 1 hidden, we should just show it)
      if min_pti > 1:
        pt_matrix.append(["...", "..."])
      elif min_pti == 1:
        # Show the 0th entry instead of "..."
        pfn = self.rng.randint(0, 2 ** self.num_bits_pfn - 1)
        if self.rng.choices([True, False], weights=[self.PROBABILITY_OF_VALID, 1 - self.PROBABILITY_OF_VALID], k=1)[0]:
          pte_val = (2 ** self.num_bits_pfn) + pfn
        else:
          pte_val = pfn
        pt_matrix.append([f"0b{0:0{self.num_bits_pti}b}", f"0b{pte_val:0{self.num_bits_pfn + 1}b}"])

      # Add actual entries
      pt_matrix.extend([
        [f"0b{pti:0{self.num_bits_pti}b}", f"0b{pte:0{self.num_bits_pfn + 1}b}"]
        for pti, pte in sorted(pt_entries.items())
      ])

      # Smart trailing ellipsis: only if there are 2+ hidden entries after
      hidden_after = max_possible_pti - max_pti
      if hidden_after > 1:
        pt_matrix.append(["...", "..."])
      elif hidden_after == 1:
        # Show the last entry instead of "..."
        pfn = self.rng.randint(0, 2 ** self.num_bits_pfn - 1)
        if self.rng.choices([True, False], weights=[self.PROBABILITY_OF_VALID, 1 - self.PROBABILITY_OF_VALID], k=1)[0]:
          pte_val = (2 ** self.num_bits_pfn) + pfn
        else:
          pte_val = pfn
        pt_matrix.append([f"0b{max_possible_pti:0{self.num_bits_pti}b}", f"0b{pte_val:0{self.num_bits_pfn + 1}b}"])

      table_group.add_table(
        label=f"PTC 0b{pt_num:0{self.num_bits_pfn}b}:",
        table=ContentAST.Table(headers=["PTI", "PTE"], data=pt_matrix)
      )

    body.add_element(table_group)

    # Answer block
    body.add_element(
      ContentAST.AnswerBlock([
        ContentAST.Answer(self.answers["answer__pdi"], label="PDI (Page Directory Index)"),
        ContentAST.Answer(self.answers["answer__pti"], label="PTI (Page Table Index)"),
        ContentAST.Answer(self.answers["answer__offset"], label="Offset"),
        ContentAST.Answer(self.answers["answer__pd_entry"], label="PD Entry (from Page Directory)"),
        ContentAST.Answer(self.answers["answer__pt_number"], label="Page Table Number"),
        ContentAST.Answer(self.answers["answer__pte"], label="PTE (from Page Table)"),
        ContentAST.Answer(self.answers["answer__is_valid"], label="VALID or INVALID?"),
        ContentAST.Answer(self.answers["answer__pfn"], label="PFN"),
        ContentAST.Answer(self.answers["answer__physical_address"], label="Physical Address"),
      ])
    )

    return body

  def get_explanation(self, *args, **kwargs) -> ContentAST.Section:
    explanation = ContentAST.Section()

    explanation.add_element(
      ContentAST.Paragraph([
        "Two-level paging requires two lookups: first in the Page Directory, then in a Page Table.",
        "The virtual address is split into three parts: PDI | PTI | Offset."
      ])
    )

    explanation.add_element(
      ContentAST.Paragraph([
        "Don't forget to pad with the appropriate number of 0s!"
      ])
    )

    # Step 1: Extract PDI, PTI, Offset
    explanation.add_element(
      ContentAST.Paragraph([
        f"<b>Step 1: Extract components from Virtual Address</b>",
        f"Virtual Address = PDI | PTI | Offset",
        f"<tt>0b{self.virtual_address:0{self.num_bits_vpn + self.num_bits_offset}b}</tt> = "
        f"<tt>0b{self.pdi:0{self.num_bits_pdi}b}</tt> | "
        f"<tt>0b{self.pti:0{self.num_bits_pti}b}</tt> | "
        f"<tt>0b{self.offset:0{self.num_bits_offset}b}</tt>"
      ])
    )

    # Step 2: Look up PD Entry
    explanation.add_element(
      ContentAST.Paragraph([
        f"<b>Step 2: Look up Page Directory Entry</b>",
        f"Using PDI = <tt>0b{self.pdi:0{self.num_bits_pdi}b}</tt>, we find PD Entry = <tt>0b{self.pd_entry:0{self.num_bits_pfn + 1}b}</tt>"
      ])
    )

    # Step 3: Check PD validity
    pd_valid_bit = self.pd_entry // (2 ** self.num_bits_pfn)
    explanation.add_element(
      ContentAST.Paragraph([
        f"<b>Step 3: Check Page Directory Entry validity</b>",
        f"The MSB (valid bit) is <b>{pd_valid_bit}</b>, so this PD Entry is <b>{'VALID' if self.pd_valid else 'INVALID'}</b>."
      ])
    )

    if not self.pd_valid:
      explanation.add_element(
        ContentAST.Paragraph([
          "Since the Page Directory Entry is invalid, the translation fails here.",
          "We write <b>INVALID</b> for all remaining fields.",
          "If it were valid, we would continue with the steps below.",
          "<hr>"
        ])
      )

    # Step 4: Extract PT number (if PD valid)
    explanation.add_element(
      ContentAST.Paragraph([
        f"<b>Step 4: Extract Page Table Number</b>",
        "We remove the valid bit from the PD Entry to get the Page Table Number:"
      ])
    )

    explanation.add_element(
      ContentAST.Equation(
        f"\\texttt{{{self.page_table_number:0{self.num_bits_pfn}b}}} = "
        f"\\texttt{{0b{self.pd_entry:0{self.num_bits_pfn + 1}b}}} \\& "
        f"\\texttt{{0b{(2 ** self.num_bits_pfn) - 1:0{self.num_bits_pfn + 1}b}}}"
      )
    )

    if self.pd_valid:
      explanation.add_element(
        ContentAST.Paragraph([
          f"This tells us to use <b>Page Table #{self.page_table_number}</b>."
        ])
      )

      # Step 5: Look up PTE
      explanation.add_element(
        ContentAST.Paragraph([
          f"<b>Step 5: Look up Page Table Entry</b>",
          f"Using PTI = <tt>0b{self.pti:0{self.num_bits_pti}b}</tt> in Page Table #{self.page_table_number}, "
          f"we find PTE = <tt>0b{self.pte:0{self.num_bits_pfn + 1}b}</tt>"
        ])
      )

      # Step 6: Check PT validity
      pt_valid_bit = self.pte // (2 ** self.num_bits_pfn)
      explanation.add_element(
        ContentAST.Paragraph([
          f"<b>Step 6: Check Page Table Entry validity</b>",
          f"The MSB (valid bit) is <b>{pt_valid_bit}</b>, so this PTE is <b>{'VALID' if self.pt_valid else 'INVALID'}</b>."
        ])
      )

      if not self.pt_valid:
        explanation.add_element(
          ContentAST.Paragraph([
            "Since the Page Table Entry is invalid, the translation fails.",
            "We write <b>INVALID</b> for PFN and Physical Address.",
            "If it were valid, we would continue with the steps below.",
            "<hr>"
          ])
        )

      # Step 7: Extract PFN
      explanation.add_element(
        ContentAST.Paragraph([
          f"<b>Step 7: Extract PFN</b>",
          "We remove the valid bit from the PTE to get the PFN:"
        ])
      )

      explanation.add_element(
        ContentAST.Equation(
          f"\\texttt{{{self.pfn:0{self.num_bits_pfn}b}}} = "
          f"\\texttt{{0b{self.pte:0{self.num_bits_pfn + 1}b}}} \\& "
          f"\\texttt{{0b{(2 ** self.num_bits_pfn) - 1:0{self.num_bits_pfn + 1}b}}}"
        )
      )

      # Step 8: Construct physical address
      explanation.add_element(
        ContentAST.Paragraph([
          f"<b>Step 8: Construct Physical Address</b>",
          "Physical Address = PFN | Offset"
        ])
      )

      explanation.add_element(
        ContentAST.Equation(
          fr"{r'\mathbf{' if self.is_valid else ''}\mathtt{{0b{self.physical_address:0{self.num_bits_pfn + self.num_bits_offset}b}}}{r'}' if self.is_valid else ''} = "
          f"\\mathtt{{0b{self.pfn:0{self.num_bits_pfn}b}}} \\mid "
          f"\\mathtt{{0b{self.offset:0{self.num_bits_offset}b}}}"
        )
      )

    return explanation
