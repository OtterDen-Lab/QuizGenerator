#!/usr/bin/env python
from __future__ import annotations

import abc
import collections
import copy
import enum
import logging
import math
import random
from typing import List

import QuizGenerator.contentast as ca
from QuizGenerator.mixins import BodyTemplatesMixin, TableQuestionMixin
from QuizGenerator.question import Question, QuestionRegistry, RegenerableChoiceMixin

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
  
  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    rng = random.Random(rng_seed)
    num_bits_va = kwargs.get("num_bits_va", rng.randint(2, cls.MAX_BITS))
    num_bits_offset = rng.randint(1, num_bits_va - 1)
    num_bits_vpn = num_bits_va - num_bits_offset

    possible_answers = {
      cls.Target.VA_BITS: ca.AnswerTypes.Int(num_bits_va, unit="bits"),
      cls.Target.OFFSET_BITS: ca.AnswerTypes.Int(num_bits_offset, unit="bits"),
      cls.Target.VPN_BITS: ca.AnswerTypes.Int(num_bits_vpn, unit="bits")
    }
    blank_kind = rng.choice(list(cls.Target))

    return {
      "num_bits_va": num_bits_va,
      "num_bits_offset": num_bits_offset,
      "num_bits_vpn": num_bits_vpn,
      "possible_answers": possible_answers,
      "blank_kind": blank_kind,
    }

  @classmethod
  def _build_body(cls, context):
    """Build question body and collect answers."""
    
    # Create table data with one blank cell
    table_data = [{}]
    for target in list(cls.Target):
      if target == context["blank_kind"]:
        # This cell should be an answer blank
        table_data[0][target.value] = context["possible_answers"][target]
      else:
        # This cell shows the value
        table_data[0][target.value] = f"{context['possible_answers'][target].display} bits"

    table = cls.create_fill_in_table(
      headers=[t.value for t in list(cls.Target)],
      template_rows=table_data
    )

    body = ca.Section()
    body.add_element(
      ca.Paragraph(
        [
          "Given the information in the below table, please complete the table as appropriate."
        ]
      )
    )
    body.add_element(table)
    return body

  @classmethod
  def _build_explanation(cls, context):
    """Build question explanation."""
    explanation = ca.Section()

    explanation.add_element(
      ca.Paragraph(
        [
          "Remember, when we are calculating the size of virtual address spaces, "
          "the number of bits in the overall address space is equal to the number of bits in the VPN "
          "plus the number of bits for the offset.",
          "We don't waste any bits!"
        ]
      )
    )

    explanation.add_element(
      ca.Paragraph(
        [
          ca.Text(f"{context['num_bits_va']}", emphasis=(context["blank_kind"] == cls.Target.VA_BITS)),
          ca.Text(" = "),
          ca.Text(f"{context['num_bits_vpn']}", emphasis=(context["blank_kind"] == cls.Target.VPN_BITS)),
          ca.Text(" + "),
          ca.Text(f"{context['num_bits_offset']}", emphasis=(context["blank_kind"] == cls.Target.OFFSET_BITS))
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
    
    self.hit_rate = 0. # placeholder

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    rng = random.Random(rng_seed)
    num_elements = kwargs.get("num_elements", 5)
    cache_size = kwargs.get("cache_size", 3)
    num_requests = kwargs.get("num_requests", 10)

    policy = kwargs.get("policy") or kwargs.get("algo")
    if policy is None:
      cache_policy = rng.choice(list(cls.Kind))
      config_params = {"policy": cache_policy.name}
    else:
      if isinstance(policy, cls.Kind):
        cache_policy = policy
      else:
        try:
          cache_policy = cls.Kind[str(policy)]
        except KeyError:
          log.warning(
            f"Invalid cache policy '{policy}'. "
            f"Valid options: {[k.name for k in cls.Kind]}. Defaulting to FIFO."
          )
          cache_policy = cls.Kind.FIFO
      config_params = {"policy": cache_policy.name}

    requests = (
        list(range(cache_size))
        + rng.choices(population=list(range(cache_size - 1)), k=1)
        + rng.choices(population=list(range(cache_size, num_elements)), k=1)
        + rng.choices(population=list(range(num_elements)), k=(num_requests - 2))
    )

    cache = cls.Cache(cache_policy, cache_size, requests)
    request_results = {}
    number_of_hits = 0
    for (request_number, request) in enumerate(requests):
      was_hit, evicted, cache_state = cache.query_cache(request, request_number)
      if was_hit:
        number_of_hits += 1
      hit_value = 'hit' if was_hit else 'miss'
      evicted_value = '-' if evicted is None else f"{evicted}"
      cache_state_value = copy.copy(cache_state)

      request_results[request_number] = {
        "request": request,
        "hit_value": hit_value,
        "evicted_value": evicted_value,
        "cache_state_value": cache_state_value,
        "hit_answer": ca.AnswerTypes.String(hit_value),
        "evicted_answer": ca.AnswerTypes.String(evicted_value),
        "cache_state_answer": ca.AnswerTypes.List(
          value=cache_state_value,
          order_matters=True
        ),
      }

    hit_rate = 100 * number_of_hits / num_requests
    hit_rate_answer = ca.AnswerTypes.Float(
      hit_rate,
      label="Hit rate, excluding non-capacity misses",
      unit="%"
    )

    return {
      "num_elements": num_elements,
      "cache_size": cache_size,
      "num_requests": num_requests,
      "cache_policy": cache_policy,
      "requests": requests,
      "request_results": request_results,
      "hit_rate": hit_rate,
      "hit_rate_answer": hit_rate_answer,
      "_config_params": config_params,
    }

  @classmethod
  def is_interesting_ctx(cls, context) -> bool:
    return (context["hit_rate"] / 100.0) < 0.7

  @classmethod
  def _build_body(cls, context):
    """Build question body and collect answers."""
    answers = []

    # Create table data for cache simulation
    table_rows = []
    for request_number in sorted(context["request_results"].keys()):
      result = context["request_results"][request_number]
      table_rows.append(
        {
          "Page Requested": f"{context['requests'][request_number]}",
          "Hit/Miss": result["hit_answer"],
          "Evicted": result["evicted_answer"],
          "Cache State": result["cache_state_answer"]
        }
      )
      # Collect answers for this request
      answers.append(result["hit_answer"])
      answers.append(result["evicted_answer"])
      answers.append(result["cache_state_answer"])

    # Create table using mixin - automatically handles answer conversion
    cache_table = cls.create_answer_table(
      headers=["Page Requested", "Hit/Miss", "Evicted", "Cache State"],
      data_rows=table_rows,
      answer_columns=["Hit/Miss", "Evicted", "Cache State"]
    )

    # Create hit rate answer block
    hit_rate_block = ca.AnswerBlock(context["hit_rate_answer"])

    # Use mixin to create complete body
    intro_text = (
      f"Assume we are using a **{context['cache_policy']}** caching policy and a cache size of **{context['cache_size']}**. "
      "Given the below series of requests please fill in the table. "
      "For the hit/miss column, please write either \"hit\" or \"miss\". "
      "For the eviction column, please write either the number of the evicted page or simply a dash (e.g. \"-\")."
    )

    instructions = ca.OnlyHtml([
      ca.Text(
        "For the cache state, please enter the cache contents in the order suggested in class, "
        "which means separated by commas with spaces (e.g. \"1, 2, 3\") "
        "and with the left-most being the next to be evicted. "
        "In the case where there is a tie, order by increasing number."
      )
    ])

    body = cls.create_fill_in_table_body(intro_text, instructions, cache_table)
    body.add_element(hit_rate_block)
    return body

  @classmethod
  def _build_explanation(cls, context):
    """Build question explanation."""
    explanation = ca.Section()

    explanation.add_element(ca.Paragraph(["The full caching table can be seen below."]))

    explanation.add_element(
      ca.Table(
        headers=["Page", "Hit/Miss", "Evicted", "Cache State"],
        data=[
          [
            context["request_results"][request]["request"],
            context["request_results"][request]["hit_value"],
            f'{context["request_results"][request]["evicted_value"]}',
            f'{",".join(map(str, context["request_results"][request]["cache_state_value"]))}',
          ]
          for (request_number, request) in enumerate(sorted(context["request_results"].keys()))
        ]
      )
    )

    explanation.add_element(
      ca.Paragraph(
        [
          "To calculate the hit rate we calculate the percentage of requests "
          "that were cache hits out of the total number of requests. "
          f"In this case we are counting only all but {context['cache_size']} requests, "
          f"since we are excluding capacity misses."
        ]
      )
    )

    return explanation


class MemoryAccessQuestion(MemoryQuestion, abc.ABC):
  PROBABILITY_OF_VALID = .875


@QuestionRegistry.register()
class BaseAndBounds(MemoryAccessQuestion, TableQuestionMixin, BodyTemplatesMixin):
  MAX_BITS = 32
  MIN_BOUNDS_BIT = 5
  MAX_BOUNDS_BITS = 16
  
  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    rng = random.Random(rng_seed)
    bounds_bits = rng.randint(
      cls.MIN_BOUNDS_BIT,
      cls.MAX_BOUNDS_BITS
    )
    base_bits = cls.MAX_BITS - bounds_bits

    bounds = int(math.pow(2, bounds_bits))
    base = rng.randint(1, int(math.pow(2, base_bits))) * bounds
    virtual_address = rng.randint(1, int(bounds / cls.PROBABILITY_OF_VALID))

    return {
      "bounds": bounds,
      "base": base,
      "virtual_address": virtual_address,
    }

  @classmethod
  def _build_body(cls, context):
    """Build question body and collect answers."""
    if context["virtual_address"] < context["bounds"]:
      answer = ca.AnswerTypes.Hex(
        context["base"] + context["virtual_address"],
        length=math.ceil(math.log2(context["base"] + context["virtual_address"]))
      )
    else:
      answer = ca.AnswerTypes.String("INVALID")

    # Use mixin to create parameter table with answer
    parameter_info = {
      "Base": f"0x{context['base']:X}",
      "Bounds": f"0x{context['bounds']:X}",
      "Virtual Address": f"0x{context['virtual_address']:X}"
    }

    table = cls.create_parameter_answer_table(
      parameter_info=parameter_info,
      answer_label="Physical Address",
      answer=answer,
      transpose=True
    )

    body = cls.create_parameter_calculation_body(
      intro_text=(
        "Given the information in the below table, "
        "please calcuate the physical address associated with the given virtual address. "
        "If the virtual address is invalid please simply write ***INVALID***."
      ),
      parameter_table=table
    )
    return body

  @classmethod
  def _build_explanation(cls, context):
    """Build question explanation."""
    explanation = ca.Section()

    explanation.add_element(
      ca.Paragraph(
        [
          "There's two steps to figuring out base and bounds.",
          "1. Are we within the bounds?\n",
          "2. If so, add to our base.\n",
          "",
        ]
      )
    )

    explanation.add_element(
      ca.Paragraph(
        [
          f"Step 1: 0x{context['virtual_address']:X} < 0x{context['bounds']:X} "
          f"--> {'***VALID***' if (context['virtual_address'] < context['bounds']) else 'INVALID'}"
        ]
      )
    )

    if context["virtual_address"] < context["bounds"]:
      explanation.add_element(
        ca.Paragraph(
          [
            f"Step 2: Since the previous check passed, we calculate "
            f"0x{context['base']:X} + 0x{context['virtual_address']:X} "
            f"= ***0x{context['base'] + context['virtual_address']:X}***.",
            "If it had been invalid we would have simply written INVALID"
          ]
        )
      )
    else:
      explanation.add_element(
        ca.Paragraph(
          [
            f"Step 2: Since the previous check failed, we simply write ***INVALID***.",
            "***If*** it had been valid, we would have calculated "
            f"0x{context['base']:X} + 0x{context['virtual_address']:X} "
            f"= 0x{context['base'] + context['virtual_address']:X}.",
          ]
        )
      )

    return explanation


@QuestionRegistry.register()
class Segmentation(MemoryAccessQuestion, TableQuestionMixin, BodyTemplatesMixin):
  MAX_BITS = 20
  MIN_VIRTUAL_BITS = 5
  MAX_VIRTUAL_BITS = 10
  
  @staticmethod
  def _within_bounds(segment, offset, bounds):
    if segment == "unallocated":
      return False
    elif bounds < offset:
      return False
    else:
      return True
  
  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    rng = random.Random(rng_seed)

    # Pick how big each of our address spaces will be
    virtual_bits = rng.randint(cls.MIN_VIRTUAL_BITS, cls.MAX_VIRTUAL_BITS)
    physical_bits = rng.randint(virtual_bits + 1, cls.MAX_BITS)

    # Start with blank base and bounds
    base = {
      "code": 0,
      "heap": 0,
      "stack": 0,
    }
    bounds = {
      "code": 0,
      "heap": 0,
      "stack": 0,
    }

    min_bounds = 4
    max_bounds = int(2 ** (virtual_bits - 2))

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

    base["unallocated"] = 0
    bounds["unallocated"] = 0

    # Make random placements and check to make sure they are not overlapping
    while (segment_collision(base, bounds)):
      for segment in base.keys():
        bounds[segment] = rng.randint(min_bounds, max_bounds - 1)
        base[segment] = rng.randint(0, (2 ** physical_bits - bounds[segment]))

    # Pick a random segment for us to use
    segment = rng.choice(list(base.keys()))
    segment_bits = {
      "code": 0,
      "heap": 1,
      "unallocated": 2,
      "stack": 3
    }[segment]

    # Try to pick a random address within that range
    if segment == "unallocated":
      offset = rng.randint(0, max_bounds - 1)
    else:
      max_offset = min(
        [
          max_bounds - 1,
          max(1, int(bounds[segment] / cls.PROBABILITY_OF_VALID))
        ]
      )
      offset = rng.randint(1, max_offset)

    # Calculate a virtual address based on the segment and the offset
    virtual_address = (
      (segment_bits << (virtual_bits - 2))
      + offset
    )

    # Calculate physical address based on offset
    physical_address = base[segment] + offset

    return {
      "virtual_bits": virtual_bits,
      "physical_bits": physical_bits,
      "base": base,
      "bounds": bounds,
      "segment": segment,
      "segment_bits": segment_bits,
      "offset": offset,
      "virtual_address": virtual_address,
      "physical_address": physical_address,
    }

  @classmethod
  def _build_body(cls, context):
    """Build question body and collect answers."""
    segment_answer = ca.AnswerTypes.String(context["segment"], label="Segment name")
    if cls._within_bounds(context["segment"], context["offset"], context["bounds"][context["segment"]]):
      physical_answer = ca.AnswerTypes.Binary(
        context["physical_address"],
        length=context["physical_bits"],
        label="Physical Address"
      )
    else:
      physical_answer = ca.AnswerTypes.String("INVALID", label="Physical Address")
    answers = [segment_answer, physical_answer]

    body = ca.Section()

    body.add_element(
      ca.Paragraph(
        [
          f"Given a virtual address space of {context['virtual_bits']}bits, "
          f"and a physical address space of {context['physical_bits']}bits, "
          "what is the physical address associated with the virtual address "
          f"0b{context['virtual_address']:0{context['virtual_bits']}b}?",
          "If it is invalid simply type INVALID.",
          "Note: assume that the stack grows in the same way as the code and the heap."
        ]
      )
    )

    # Create segment table using mixin
    segment_rows = [
      {
        "": "code",
        "base": f"0b{context['base']['code']:0{context['physical_bits']}b}",
        "bounds": f"0b{context['bounds']['code']:0b}"
      },
      {
        "": "heap",
        "base": f"0b{context['base']['heap']:0{context['physical_bits']}b}",
        "bounds": f"0b{context['bounds']['heap']:0b}"
      },
      {
        "": "stack",
        "base": f"0b{context['base']['stack']:0{context['physical_bits']}b}",
        "bounds": f"0b{context['bounds']['stack']:0b}"
      }
    ]

    segment_table = cls.create_answer_table(
      headers=["", "base", "bounds"],
      data_rows=segment_rows,
      answer_columns=[]  # No answer columns in this table
    )

    body.add_element(segment_table)

    body.add_element(
      ca.AnswerBlock([
        segment_answer,
        physical_answer
      ])
    )
    return body, answers

  @classmethod
  def _build_explanation(cls, context):
    explanation = ca.Section()
    
    explanation.add_element(
      ca.Paragraph(
        [
          "The core idea to keep in mind with segmentation is that you should always check ",
          "the first two bits of the virtual address to see what segment it is in and then go from there."
          "Keep in mind, "
          "we also may need to include padding if our virtual address has a number of leading zeros left off!"
        ]
      )
    )
    
    explanation.add_element(
      ca.Paragraph(
        [
          f"In this problem our virtual address, "
          f"converted to binary and including padding, is 0b{context['virtual_address']:0{context['virtual_bits']}b}.",
          f"From this we know that our segment bits are 0b{context['segment_bits']:02b}, "
          f"meaning that we are in the ***{context['segment']}*** segment.",
          ""
        ]
      )
    )
    
    if context["segment"] == "unallocated":
      explanation.add_element(
        ca.Paragraph(
          [
            "Since this is the unallocated segment there are no possible valid translations, so we enter ***INVALID***."
          ]
        )
      )
    else:
      explanation.add_element(
        ca.Paragraph(
          [
            f"Since we are in the {context['segment']} segment, "
            f"we see from our table that our bounds are {context['bounds'][context['segment']]}. "
            f"Remember that our check for our {context['segment']} segment is: ",
            f"`if (offset >= bounds({context['segment']})) : INVALID`",
            "which becomes"
            f"`if ({context['offset']:0b} > {context['bounds'][context['segment']]:0b}) : INVALID`"
          ]
        )
      )
      
      if not cls._within_bounds(context["segment"], context["offset"], context["bounds"][context["segment"]]):
        # then we are outside of bounds
        explanation.add_element(
          ca.Paragraph(
            [
              "We can therefore see that we are outside of bounds so we should put ***INVALID***.",
              "If we <i>were</i> requesting a valid memory location we could use the below steps to do so."
              "<hr>"
            ]
          )
        )
      else:
        explanation.add_element(
          ca.Paragraph(
            [
              "We are therefore in bounds so we can calculate our physical address, as we do below."
            ]
          )
        )
      
      explanation.add_element(
        ca.Paragraph(
          [
            "To find the physical address we use the formula:",
            "<code>physical_address = base(segment) + offset</code>",
            "which becomes",
            f"<code>physical_address = {context['base'][context['segment']]:0b} + {context['offset']:0b}</code>.",
            ""
          ]
        )
      )
      
      explanation.add_element(
        ca.Paragraph(
          [
            "Lining this up for ease we can do this calculation as:"
          ]
        )
      )
      explanation.add_element(
        ca.Code(
          f"  0b{context['base'][context['segment']]:0{context['physical_bits']}b}\n"
          f"<u>+ 0b{context['offset']:0{context['physical_bits']}b}</u>\n"
          f"  0b{context['physical_address']:0{context['physical_bits']}b}\n"
        )
      )

    return explanation, []


@QuestionRegistry.register()
class Paging(MemoryAccessQuestion, TableQuestionMixin, BodyTemplatesMixin):
  MIN_OFFSET_BITS = 3
  MIN_VPN_BITS = 3
  MIN_PFN_BITS = 3
  
  MAX_OFFSET_BITS = 8
  MAX_VPN_BITS = 8
  MAX_PFN_BITS = 16
  
  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    rng = random.Random(rng_seed)

    num_bits_offset = rng.randint(cls.MIN_OFFSET_BITS, cls.MAX_OFFSET_BITS)
    num_bits_vpn = rng.randint(cls.MIN_VPN_BITS, cls.MAX_VPN_BITS)
    num_bits_pfn = rng.randint(max([cls.MIN_PFN_BITS, num_bits_vpn]), cls.MAX_PFN_BITS)

    virtual_address = rng.randint(1, 2 ** (num_bits_vpn + num_bits_offset))

    # Calculate these two
    offset = virtual_address % (2 ** (num_bits_offset))
    vpn = virtual_address // (2 ** (num_bits_offset))

    # Generate this randomly
    pfn = rng.randint(0, 2 ** (num_bits_pfn))

    # Calculate this
    physical_address = pfn * (2 ** num_bits_offset) + offset

    if rng.choices([True, False], weights=[(cls.PROBABILITY_OF_VALID), (1 - cls.PROBABILITY_OF_VALID)], k=1)[0]:
      is_valid = True
      # Set our actual entry to be in the table and valid
      pte = pfn + (2 ** (num_bits_pfn))
    else:
      is_valid = False
      # Leave it as invalid
      pte = pfn

    # Generate page table (moved from get_body to ensure deterministic generation)
    table_size = rng.randint(5, 8)

    lowest_possible_bottom = max([0, vpn - table_size])
    highest_possible_bottom = min([2 ** num_bits_vpn - table_size, vpn])

    table_bottom = rng.randint(lowest_possible_bottom, highest_possible_bottom)
    table_top = table_bottom + table_size

    page_table = {}
    page_table[vpn] = pte

    # Fill in the rest of the table
    for vpn_idx in range(table_bottom, table_top):
      if vpn_idx == vpn:
        continue
      pte_candidate = page_table[vpn]
      while pte_candidate in page_table.values():
        pte_candidate = rng.randint(0, 2 ** num_bits_pfn - 1)
        if rng.choices([True, False], weights=[(1 - cls.PROBABILITY_OF_VALID), cls.PROBABILITY_OF_VALID], k=1)[0]:
          # Randomly set it to be valid
          pte_candidate += (2 ** num_bits_pfn)
      # Once we have a unique random entry, put it into the Page Table
      page_table[vpn_idx] = pte_candidate

    return {
      'num_bits_offset': num_bits_offset,
      'num_bits_vpn': num_bits_vpn,
      'num_bits_pfn': num_bits_pfn,
      'virtual_address': virtual_address,
      'offset': offset,
      'vpn': vpn,
      'pfn': pfn,
      'physical_address': physical_address,
      'is_valid': is_valid,
      'pte': pte,
      'page_table': page_table,
    }

  @classmethod
  def _build_body(cls, context):
    """Build question body and collect answers."""
    vpn_answer = ca.AnswerTypes.Binary(context['vpn'], length=context['num_bits_vpn'], label='VPN')
    offset_answer = ca.AnswerTypes.Binary(context['offset'], length=context['num_bits_offset'], label='Offset')
    pte_answer = ca.AnswerTypes.Binary(context['pte'], length=(context['num_bits_pfn'] + 1), label='PTE')
    if context['is_valid']:
      is_valid_answer = ca.AnswerTypes.String('VALID', label='VALID or INVALID?')
      pfn_answer = ca.AnswerTypes.Binary(context['pfn'], length=context['num_bits_pfn'], label='PFN')
      physical_answer = ca.AnswerTypes.Binary(
        context['physical_address'],
        length=(context['num_bits_pfn'] + context['num_bits_offset']),
        label='Physical Address'
      )
    else:
      is_valid_answer = ca.AnswerTypes.String('INVALID', label='VALID or INVALID?')
      pfn_answer = ca.AnswerTypes.String('INVALID', label='PFN')
      physical_answer = ca.AnswerTypes.String('INVALID', label='Physical Address')

    answers = [
      vpn_answer,
      offset_answer,
      pte_answer,
      is_valid_answer,
      pfn_answer,
      physical_answer,
    ]

    body = ca.Section()

    body.add_element(
      ca.Paragraph(
        [
          'Given the below information please calculate the equivalent physical address of the given virtual address, filling out all steps along the way.',
          'Remember, we typically have the MSB representing valid or invalid.'
        ]
      )
    )

    # Create parameter info table using mixin
    parameter_info = {
      'Virtual Address': f"0b{context['virtual_address']:0{context['num_bits_vpn'] + context['num_bits_offset']}b}",
      '# VPN bits': f"{context['num_bits_vpn']}",
      '# PFN bits': f"{context['num_bits_pfn']}"
    }

    body.add_element(cls.create_info_table(parameter_info))

    # Use the page table generated in _build_context for deterministic output
    # Add in ellipses before and after page table entries, if appropriate
    value_matrix = []

    if min(context['page_table'].keys()) != 0:
      value_matrix.append(['...', '...'])

    value_matrix.extend(
      [
        [f"0b{vpn:0{context['num_bits_vpn']}b}", f"0b{pte:0{(context['num_bits_pfn'] + 1)}b}"]
        for vpn, pte in sorted(context['page_table'].items())
      ]
    )

    if (max(context['page_table'].keys()) + 1) != 2 ** context['num_bits_vpn']:
      value_matrix.append(['...', '...'])

    body.add_element(
      ca.Table(
        headers=['VPN', 'PTE'],
        data=value_matrix
      )
    )

    body.add_element(
      ca.AnswerBlock([
        vpn_answer,
        offset_answer,
        pte_answer,
        is_valid_answer,
        pfn_answer,
        physical_answer,
      ])
    )

    return body, answers
  
  @classmethod
  def _build_explanation(cls, context):
    """Build question explanation."""
    explanation = ca.Section()

    explanation.add_element(
      ca.Paragraph(
        [
          'The core idea of Paging is we want to break the virtual address into the VPN and the offset.  '
          'From here, we get the Page Table Entry corresponding to the VPN, and check the validity of the entry.  '
          'If it is valid, we clear the metadata and attach the PFN to the offset and have our physical address.',
        ]
      )
    )

    explanation.add_element(
      ca.Paragraph(
        [
          "Don't forget to pad with the appropriate number of 0s (the appropriate number is the number of bits)!",
        ]
      )
    )

    explanation.add_element(
      ca.Paragraph(
        [
          'Virtual Address = VPN | offset',
          f"<tt>0b{context['virtual_address']:0{context['num_bits_vpn'] + context['num_bits_offset']}b}</tt> "
          f"= <tt>0b{context['vpn']:0{context['num_bits_vpn']}b}</tt> | <tt>0b{context['offset']:0{context['num_bits_offset']}b}</tt>",
        ]
      )
    )

    explanation.add_element(
      ca.Paragraph(
        [
          'We next use our VPN to index into our page table and find the corresponding entry.'
          'Our Page Table Entry is ',
          f"<tt>0b{context['pte']:0{(context['num_bits_pfn'] + 1)}b}</tt>"
          'which we found by looking for our VPN in the page table.',
        ]
      )
    )

    if context['is_valid']:
      explanation.add_element(
        ca.Paragraph(
          [
            f"In our PTE we see that the first bit is **{context['pte'] // (2 ** context['num_bits_pfn'])}** meaning that the translation is **VALID**"
          ]
        )
      )
    else:
      explanation.add_element(
        ca.Paragraph(
          [
            f"In our PTE we see that the first bit is **{context['pte'] // (2 ** context['num_bits_pfn'])}** meaning that the translation is **INVALID**.",
            'Therefore, we just write "INVALID" as our answer.',
            'If it were valid we would complete the below steps.',
            '<hr>'
          ]
        )
      )

    explanation.add_element(
      ca.Paragraph(
        [
          'Next, we convert our PTE to our PFN by removing our metadata.  '
          "In this case we are just removing the leading bit.  We can do this by applying a binary mask.",
          'PFN = PTE & mask',
          'which is,',
        ]
      )
    )
    explanation.add_element(
      ca.Equation(
        f"\\texttt{{{context['pfn']:0{context['num_bits_pfn']}b}}} "
        f"= \\texttt{{0b{context['pte']:0{context['num_bits_pfn'] + 1}b}}} "
        f"\\& \\texttt{{0b{(2 ** context['num_bits_pfn']) - 1:0{context['num_bits_pfn'] + 1}b}}}"
      )
    )

    explanation.add_elements(
      [
        ca.Paragraph(
          [
            'We then add combine our PFN and offset, '
            'Physical Address = PFN | offset',
          ]
        ),
        ca.Equation(
          fr"{r'\mathbf{' if context['is_valid'] else ''}\mathtt{{0b{context['physical_address']:0{context['num_bits_pfn'] + context['num_bits_offset']}b}}}{r'}' if context['is_valid'] else ''} = \mathtt{{0b{context['pfn']:0{context['num_bits_pfn']}b}}} \mid \mathtt{{0b{context['offset']:0{context['num_bits_offset']}b}}}"
        )
      ]
    )

    explanation.add_elements(
      [
        ca.Paragraph(['Note: Strictly speaking, this calculation is:', ]),
        ca.Equation(
          fr"{r'\mathbf{' if context['is_valid'] else ''}\mathtt{{0b{context['physical_address']:0{context['num_bits_pfn'] + context['num_bits_offset']}b}}}{r'}' if context['is_valid'] else ''} = \mathtt{{0b{context['pfn']:0{context['num_bits_pfn']}b}{0:0{context['num_bits_offset']}}}} + \mathtt{{0b{context['offset']:0{context['num_bits_offset']}b}}}"
        ),
        ca.Paragraph(["But that's a lot of extra 0s, so I'm splitting them up for succinctness"])
      ]
    )

    return explanation, []


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

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    rng = random.Random(rng_seed)

    # Set up bit counts
    num_bits_offset = rng.randint(cls.MIN_OFFSET_BITS, cls.MAX_OFFSET_BITS)
    num_bits_pdi = rng.randint(cls.MIN_PDI_BITS, cls.MAX_PDI_BITS)
    num_bits_pti = rng.randint(cls.MIN_PTI_BITS, cls.MAX_PTI_BITS)
    num_bits_pfn = rng.randint(cls.MIN_PFN_BITS, cls.MAX_PFN_BITS)

    # Total VPN bits = PDI + PTI
    num_bits_vpn = num_bits_pdi + num_bits_pti

    # Generate a random virtual address
    virtual_address = rng.randint(1, 2 ** (num_bits_vpn + num_bits_offset))

    # Extract components from virtual address
    offset = virtual_address % (2 ** num_bits_offset)
    vpn = virtual_address // (2 ** num_bits_offset)

    pti = vpn % (2 ** num_bits_pti)
    pdi = vpn // (2 ** num_bits_pti)

    # Generate PFN randomly
    pfn = rng.randint(0, 2 ** num_bits_pfn - 1)

    # Calculate physical address
    physical_address = pfn * (2 ** num_bits_offset) + offset

    # Determine validity at both levels
    # PD entry can be valid or invalid
    pd_valid = rng.choices([True, False], weights=[cls.PROBABILITY_OF_VALID, 1 - cls.PROBABILITY_OF_VALID], k=1)[0]

    # PT entry only matters if PD is valid
    if pd_valid:
      pt_valid = rng.choices([True, False], weights=[cls.PROBABILITY_OF_VALID, 1 - cls.PROBABILITY_OF_VALID], k=1)[0]
    else:
      pt_valid = False  # Doesn't matter, won't be checked

    # Generate a page table number (PTBR - Page Table Base Register value in the PD entry)
    # This represents which page table to use
    page_table_number = rng.randint(0, 2 ** num_bits_pfn - 1)

    # Create PD entry: valid bit + page table number
    if pd_valid:
      pd_entry = (2 ** num_bits_pfn) + page_table_number
    else:
      pd_entry = page_table_number  # Invalid, no valid bit set

    # Create PT entry: valid bit + PFN
    if pt_valid:
      pte = (2 ** num_bits_pfn) + pfn
    else:
      pte = pfn  # Invalid, no valid bit set

    # Overall validity requires both levels to be valid
    is_valid = pd_valid and pt_valid

    # Build page directory - show 3-4 entries
    pd_size = rng.randint(3, 4)
    lowest_pd_bottom = max([0, pdi - pd_size])
    highest_pd_bottom = min([2 ** num_bits_pdi - pd_size, pdi])
    pd_bottom = rng.randint(lowest_pd_bottom, highest_pd_bottom)
    pd_top = pd_bottom + pd_size

    page_directory = {}
    page_directory[pdi] = pd_entry

    # Fill in other PD entries
    for pdi_idx in range(pd_bottom, pd_top):
      if pdi_idx == pdi:
        continue
      # Generate random PD entry
      pt_num = rng.randint(0, 2 ** num_bits_pfn - 1)
      while pt_num == page_table_number:  # Make sure it's different
        pt_num = rng.randint(0, 2 ** num_bits_pfn - 1)

      # Randomly valid or invalid
      if rng.choices([True, False], weights=[cls.PROBABILITY_OF_VALID, 1 - cls.PROBABILITY_OF_VALID], k=1)[0]:
        pd_val = (2 ** num_bits_pfn) + pt_num
      else:
        pd_val = pt_num

      page_directory[pdi_idx] = pd_val

    # Build 2-3 page tables to show
    # Always include the one we need, plus 1-2 others
    num_page_tables_to_show = rng.randint(2, 3)

    # Get unique page table numbers from the PD entries (extract PT numbers from valid entries)
    shown_pt_numbers = set()
    for pd_val in page_directory.values():
      pt_num = pd_val % (2 ** num_bits_pfn)  # Extract PT number (remove valid bit)
      shown_pt_numbers.add(pt_num)

    # Ensure our required page table is included
    shown_pt_numbers.add(page_table_number)

    # Limit to requested number, but ALWAYS keep the required page table
    shown_pt_numbers_list = list(shown_pt_numbers)
    if page_table_number in shown_pt_numbers_list:
      # Remove it temporarily so we can add it back first
      shown_pt_numbers_list.remove(page_table_number)
    # Start with required page table, then add others up to the limit
    shown_pt_numbers = [page_table_number] + shown_pt_numbers_list[:num_page_tables_to_show - 1]

    # Build each page table
    page_tables = {}  # Dict mapping PT number -> dict of PTI -> PTE

    # Use consistent size for all page tables for cleaner presentation
    pt_size = rng.randint(2, 4)

    # Determine the PTI range that all tables will use (based on target PTI)
    # This ensures all tables show the same PTI values for consistency
    lowest_pt_bottom = max([0, pti - pt_size + 1])
    highest_pt_bottom = min([2 ** num_bits_pti - pt_size, pti])
    pt_bottom = rng.randint(lowest_pt_bottom, highest_pt_bottom)
    pt_top = pt_bottom + pt_size

    # Generate all page tables using the SAME PTI range
    for pt_num in shown_pt_numbers:
      page_tables[pt_num] = {}

      for pti_idx in range(pt_bottom, pt_top):
        if pt_num == page_table_number and pti_idx == pti:
          # Use the actual answer for the target page table entry
          page_tables[pt_num][pti_idx] = pte
        else:
          # Generate random PTE for all other entries
          pfn_rand = rng.randint(0, 2 ** num_bits_pfn - 1)
          if rng.choices([True, False], weights=[cls.PROBABILITY_OF_VALID, 1 - cls.PROBABILITY_OF_VALID], k=1)[0]:
            pte_val = (2 ** num_bits_pfn) + pfn_rand
          else:
            pte_val = pfn_rand

          page_tables[pt_num][pti_idx] = pte_val

    def random_pte_value():
      pfn_rand = rng.randint(0, 2 ** num_bits_pfn - 1)
      if rng.choices([True, False], weights=[cls.PROBABILITY_OF_VALID, 1 - cls.PROBABILITY_OF_VALID], k=1)[0]:
        return (2 ** num_bits_pfn) + pfn_rand
      return pfn_rand

    pt_display_extras = {}
    for pt_num, pt_entries in page_tables.items():
      min_pti = min(pt_entries.keys())
      max_pti = max(pt_entries.keys())
      max_possible_pti = 2 ** num_bits_pti - 1
      leading = None
      trailing = None
      if min_pti == 1:
        leading = (0, random_pte_value())
      if (max_possible_pti - max_pti) == 1:
        trailing = (max_possible_pti, random_pte_value())
      pt_display_extras[pt_num] = {
        'leading': leading,
        'trailing': trailing,
      }

    return {
      'num_bits_offset': num_bits_offset,
      'num_bits_pdi': num_bits_pdi,
      'num_bits_pti': num_bits_pti,
      'num_bits_pfn': num_bits_pfn,
      'num_bits_vpn': num_bits_vpn,
      'virtual_address': virtual_address,
      'offset': offset,
      'pdi': pdi,
      'pti': pti,
      'pfn': pfn,
      'physical_address': physical_address,
      'pd_valid': pd_valid,
      'pt_valid': pt_valid,
      'page_table_number': page_table_number,
      'pd_entry': pd_entry,
      'pte': pte,
      'is_valid': is_valid,
      'page_directory': page_directory,
      'page_tables': page_tables,
      'pt_display_extras': pt_display_extras,
    }

  @classmethod
  def _build_body(cls, context):
    """Build question body and collect answers."""
    pdi_answer = ca.AnswerTypes.Binary(context['pdi'], length=context['num_bits_pdi'], label='PDI (Page Directory Index)')
    pti_answer = ca.AnswerTypes.Binary(context['pti'], length=context['num_bits_pti'], label='PTI (Page Table Index)')
    offset_answer = ca.AnswerTypes.Binary(context['offset'], length=context['num_bits_offset'], label='Offset')
    pd_entry_answer = ca.AnswerTypes.Binary(context['pd_entry'], length=(context['num_bits_pfn'] + 1), label='PD Entry (from Page Directory)')
    if context['pd_valid']:
      pt_number_answer = ca.AnswerTypes.Binary(context['page_table_number'], length=context['num_bits_pfn'], label='Page Table Number')
      pte_answer = ca.AnswerTypes.Binary(context['pte'], length=(context['num_bits_pfn'] + 1), label='PTE (from Page Table)')
    else:
      pt_number_answer = ca.AnswerTypes.String('INVALID', label='Page Table Number')
      pte_answer = ca.AnswerTypes.String(['INVALID', 'N/A'], label='PTE (from Page Table)')

    if context['pd_valid'] and context['pt_valid']:
      is_valid_answer = ca.AnswerTypes.String('VALID', label='VALID or INVALID?')
      pfn_answer = ca.AnswerTypes.Binary(context['pfn'], length=context['num_bits_pfn'], label='PFN')
      physical_answer = ca.AnswerTypes.Binary(
        context['physical_address'],
        length=(context['num_bits_pfn'] + context['num_bits_offset']),
        label='Physical Address'
      )
    else:
      is_valid_answer = ca.AnswerTypes.String('INVALID', label='VALID or INVALID?')
      pfn_answer = ca.AnswerTypes.String('INVALID', label='PFN')
      physical_answer = ca.AnswerTypes.String('INVALID', label='Physical Address')

    answers = [
      pdi_answer,
      pti_answer,
      offset_answer,
      pd_entry_answer,
      pt_number_answer,
      pte_answer,
      is_valid_answer,
      pfn_answer,
      physical_answer,
    ]

    body = ca.Section()

    body.add_element(
      ca.Paragraph([
        'Given the below information please calculate the equivalent physical address of the given virtual address, filling out all steps along the way.',
        'This problem uses **two-level (hierarchical) paging**.',
        'Remember, we typically have the MSB representing valid or invalid.'
      ])
    )

    # Create parameter info table using mixin (same format as Paging question)
    parameter_info = {
      'Virtual Address': f"0b{context['virtual_address']:0{context['num_bits_vpn'] + context['num_bits_offset']}b}",
      '# PDI bits': f"{context['num_bits_pdi']}",
      '# PTI bits': f"{context['num_bits_pti']}",
      '# Offset bits': f"{context['num_bits_offset']}",
      '# PFN bits': f"{context['num_bits_pfn']}"
    }

    body.add_element(cls.create_info_table(parameter_info))

    # Page Directory table
    pd_matrix = []
    if min(context['page_directory'].keys()) != 0:
      pd_matrix.append(['...', '...'])

    pd_matrix.extend([
      [f"0b{pdi:0{context['num_bits_pdi']}b}", f"0b{pd_val:0{context['num_bits_pfn'] + 1}b}"]
      for pdi, pd_val in sorted(context['page_directory'].items())
    ])

    if (max(context['page_directory'].keys()) + 1) != 2 ** context['num_bits_pdi']:
      pd_matrix.append(['...', '...'])

    # Use a simple text paragraph - the bold will come from markdown conversion
    body.add_element(
      ca.Paragraph([
        '**Page Directory:**'
      ])
    )
    body.add_element(
      ca.Table(
        headers=['PDI', 'PD Entry'],
        data=pd_matrix
      )
    )

    # Page Tables - use TableGroup for side-by-side display
    table_group = ca.TableGroup()

    for pt_num in sorted(context['page_tables'].keys()):
      pt_matrix = []
      pt_entries = context['page_tables'][pt_num]

      min_pti = min(pt_entries.keys())
      max_pti = max(pt_entries.keys())
      max_possible_pti = 2 ** context['num_bits_pti'] - 1

      # Smart leading ellipsis: only if there are 2+ hidden entries before
      # (if only 1 hidden, we should just show it)
      if min_pti > 1:
        pt_matrix.append(['...', '...'])
      elif min_pti == 1:
        leading = context['pt_display_extras'][pt_num]['leading']
        if leading is not None:
          leading_pti, leading_pte = leading
          pt_matrix.append([f"0b{leading_pti:0{context['num_bits_pti']}b}", f"0b{leading_pte:0{context['num_bits_pfn'] + 1}b}"])

      # Add actual entries
      pt_matrix.extend([
        [f"0b{pti:0{context['num_bits_pti']}b}", f"0b{pte:0{context['num_bits_pfn'] + 1}b}"]
        for pti, pte in sorted(pt_entries.items())
      ])

      # Smart trailing ellipsis: only if there are 2+ hidden entries after
      hidden_after = max_possible_pti - max_pti
      if hidden_after > 1:
        pt_matrix.append(['...', '...'])
      elif hidden_after == 1:
        trailing = context['pt_display_extras'][pt_num]['trailing']
        if trailing is not None:
          trailing_pti, trailing_pte = trailing
          pt_matrix.append([f"0b{trailing_pti:0{context['num_bits_pti']}b}", f"0b{trailing_pte:0{context['num_bits_pfn'] + 1}b}"])

      table_group.add_table(
        label=f"PTC 0b{pt_num:0{context['num_bits_pfn']}b}:",
        table=ca.Table(headers=['PTI', 'PTE'], data=pt_matrix)
      )

    body.add_element(table_group)

    # Answer block
    body.add_element(
      ca.AnswerBlock([
        pdi_answer,
        pti_answer,
        offset_answer,
        pd_entry_answer,
        pt_number_answer,
        pte_answer,
        is_valid_answer,
        pfn_answer,
        physical_answer,
      ])
    )

    return body, answers

  @classmethod
  def _build_explanation(cls, context):
    """Build question explanation."""
    explanation = ca.Section()

    explanation.add_element(
      ca.Paragraph([
        'Two-level paging requires two lookups: first in the Page Directory, then in a Page Table.',
        'The virtual address is split into three parts: PDI | PTI | Offset.'
      ])
    )

    explanation.add_element(
      ca.Paragraph([
        "Don't forget to pad with the appropriate number of 0s!"
      ])
    )

    # Step 1: Extract PDI, PTI, Offset
    explanation.add_element(
      ca.Paragraph([
        '**Step 1: Extract components from Virtual Address**',
        'Virtual Address = PDI | PTI | Offset',
        f"<tt>0b{context['virtual_address']:0{context['num_bits_vpn'] + context['num_bits_offset']}b}</tt> = "
        f"<tt>0b{context['pdi']:0{context['num_bits_pdi']}b}</tt> | "
        f"<tt>0b{context['pti']:0{context['num_bits_pti']}b}</tt> | "
        f"<tt>0b{context['offset']:0{context['num_bits_offset']}b}</tt>"
      ])
    )

    # Step 2: Look up PD Entry
    explanation.add_element(
      ca.Paragraph([
        '**Step 2: Look up Page Directory Entry**',
        f"Using PDI = <tt>0b{context['pdi']:0{context['num_bits_pdi']}b}</tt>, we find PD Entry = <tt>0b{context['pd_entry']:0{context['num_bits_pfn'] + 1}b}</tt>"
      ])
    )

    # Step 3: Check PD validity
    pd_valid_bit = context['pd_entry'] // (2 ** context['num_bits_pfn'])
    explanation.add_element(
      ca.Paragraph([
        '**Step 3: Check Page Directory Entry validity**',
        f"The MSB (valid bit) is **{pd_valid_bit}**, so this PD Entry is **{'VALID' if context['pd_valid'] else 'INVALID'}**."
      ])
    )

    if not context['pd_valid']:
      explanation.add_element(
        ca.Paragraph([
          'Since the Page Directory Entry is invalid, the translation fails here.',
          'We write **INVALID** for all remaining fields.',
          'If it were valid, we would continue with the steps below.',
          '<hr>'
        ])
      )

    # Step 4: Extract PT number (if PD valid)
    explanation.add_element(
      ca.Paragraph([
        '**Step 4: Extract Page Table Number**',
        'We remove the valid bit from the PD Entry to get the Page Table Number:'
      ])
    )

    explanation.add_element(
      ca.Equation(
        f"\\texttt{{{context['page_table_number']:0{context['num_bits_pfn']}b}}} = "
        f"\\texttt{{0b{context['pd_entry']:0{context['num_bits_pfn'] + 1}b}}} \\& "
        f"\\texttt{{0b{(2 ** context['num_bits_pfn']) - 1:0{context['num_bits_pfn'] + 1}b}}}"
      )
    )

    if context['pd_valid']:
      explanation.add_element(
        ca.Paragraph([
          f"This tells us to use **Page Table #{context['page_table_number']}**."
        ])
      )

      # Step 5: Look up PTE
      explanation.add_element(
        ca.Paragraph([
          '**Step 5: Look up Page Table Entry**',
          f"Using PTI = <tt>0b{context['pti']:0{context['num_bits_pti']}b}</tt> in Page Table #{context['page_table_number']}, "
          f"we find PTE = <tt>0b{context['pte']:0{context['num_bits_pfn'] + 1}b}</tt>"
        ])
      )

      # Step 6: Check PT validity
      pt_valid_bit = context['pte'] // (2 ** context['num_bits_pfn'])
      explanation.add_element(
        ca.Paragraph([
          '**Step 6: Check Page Table Entry validity**',
          f"The MSB (valid bit) is **{pt_valid_bit}**, so this PTE is **{'VALID' if context['pt_valid'] else 'INVALID'}**."
        ])
      )

      if not context['pt_valid']:
        explanation.add_element(
          ca.Paragraph([
            'Since the Page Table Entry is invalid, the translation fails.',
            'We write **INVALID** for PFN and Physical Address.',
            'If it were valid, we would continue with the steps below.',
            '<hr>'
          ])
        )

      # Step 7: Extract PFN
      explanation.add_element(
        ca.Paragraph([
          '**Step 7: Extract PFN**',
          'We remove the valid bit from the PTE to get the PFN:'
        ])
      )

      explanation.add_element(
        ca.Equation(
          f"\\texttt{{{context['pfn']:0{context['num_bits_pfn']}b}}} = "
          f"\\texttt{{0b{context['pte']:0{context['num_bits_pfn'] + 1}b}}} \\& "
          f"\\texttt{{0b{(2 ** context['num_bits_pfn']) - 1:0{context['num_bits_pfn'] + 1}b}}}"
        )
      )

      # Step 8: Construct physical address
      explanation.add_element(
        ca.Paragraph([
          '**Step 8: Construct Physical Address**',
          'Physical Address = PFN | Offset'
        ])
      )

      explanation.add_element(
        ca.Equation(
          fr"{r'\mathbf{' if context['is_valid'] else ''}\mathtt{{0b{context['physical_address']:0{context['num_bits_pfn'] + context['num_bits_offset']}b}}}{r'}' if context['is_valid'] else ''} = "
        rf"\mathtt{{0b{context['pfn']:0{context['num_bits_pfn']}b}}} \mid "
        rf"\mathtt{{0b{context['offset']:0{context['num_bits_offset']}b}}}"
        )
      )

    return explanation, []
