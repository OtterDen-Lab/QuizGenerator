#!env python
from __future__ import annotations

import abc
import enum
import logging
import random
from typing import Dict, List, Optional

import QuizGenerator.contentast as ca
from QuizGenerator.question import Question, QuestionRegistry

log = logging.getLogger(__name__)


class LanguageQuestion(Question, abc.ABC):
  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.LANGUAGES)
    super().__init__(*args, **kwargs)

class BNF:
  
  class Grammar:
    def __init__(self, symbols, start_symbol=None):
      self.start_symbol = start_symbol if start_symbol is not None else symbols[0]
      self.symbols = symbols
  
    def generate(self, include_spaces=False, early_exit=False, early_exit_min_iterations=5):
      curr_symbols : List[BNF.Symbol] = [self.start_symbol]
      prev_symbols: List[BNF.Symbol] = curr_symbols
      
      iteration_count = 0
      # Check to see if we have any non-terminals left
      while any(map(lambda s: s.kind == BNF.Symbol.Kind.NonTerminal, curr_symbols)):
        # Grab the previous symbols in case we are early exitting
        prev_symbols = curr_symbols
        
        # Walk through the current symbols and build a new list of symbols from it
        next_symbols : List[BNF.Symbol] = []
        for symbol in curr_symbols:
          next_symbols.extend(symbol.expand())
        curr_symbols = next_symbols
        
        iteration_count += 1
        
        if early_exit and iteration_count > early_exit_min_iterations:
          break
        
      if early_exit:
        # If we are doing an early exit then we are going to return things with non-terminals
        curr_symbols = prev_symbols
      
      # Take all the current symbols and combine them
      return ('' if not include_spaces else ' ').join([str(s) for s in curr_symbols])
    
    def print(self):
      for symbol in self.symbols:
        print(symbol.get_full_str())
        
    def get_grammar_string(self):
      lines = []
      for symbol in self.symbols:
        lines.append(f"{symbol.get_full_str()}")

      return '\n'.join(lines)
      
  class Symbol:
    
    class Kind(enum.Enum):
      NonTerminal = enum.auto()
      Terminal = enum.auto()
      
    def __init__(self, symbol : str, kind : Kind, rng):
      self.symbol = symbol
      self.kind = kind
      self.productions : List[BNF.Production] = [] # productions
      self.rng = rng
    
    def __str__(self):
      # if self.kind == BNF.Symbol.Kind.NonTerminal:
      #   return f"`{self.symbol}`"
      # else:
      #   return f"{self.symbol}"
      return f"{self.symbol}"
    
    def get_full_str(self):
      return f"{self.symbol} ::= {' | '.join([str(p) for p in self.productions])}"
    
    def add_production(self, production: BNF.Production):
      self.productions.append(production)
    
    def expand(self) -> List[BNF.Symbol]:
      if self.kind == BNF.Symbol.Kind.Terminal:
        return [self]
      return self.rng.choice(self.productions).production
  
  class Production:
    def __init__(self, production_line, nonterminal_symbols: Dict[str, BNF.Symbol], rng):
      if len(production_line.strip()) == 0:
        self.production = []
      else:
        self.production = [
          (nonterminal_symbols.get(symbol, BNF.Symbol(symbol, BNF.Symbol.Kind.Terminal, rng=rng)))
          for symbol in production_line.split(' ')
        ]
      
    def __str__(self):
      if len(self.production) == 0:
        return '""'
      return f"{' '.join([str(s) for s in self.production])}"
  
  
  @staticmethod
  def parse_bnf(grammar_str, rng) -> BNF.Grammar:
    
    # Figure out all the nonterminals and create a Token for them
    terminal_symbols = {}
    start_symbol = None
    for line in grammar_str.strip().splitlines():
      if "::=" in line:
        non_terminal_str, _ = line.split("::=", 1)
        non_terminal_str = non_terminal_str.strip()
        
        terminal_symbols[non_terminal_str] = BNF.Symbol(non_terminal_str, BNF.Symbol.Kind.NonTerminal, rng=rng)
        if start_symbol is None:
          start_symbol = terminal_symbols[non_terminal_str]
    
    # Parse the grammar statement
    for line in grammar_str.strip().splitlines():
      if "::=" in line:
        # Split the line into non-terminal and its expansions
        non_terminal_str, expansions = line.split("::=", 1)
        non_terminal_str = non_terminal_str.strip()
        
        non_terminal = terminal_symbols[non_terminal_str]
        
        for production_str in expansions.split('|'):
          production_str = production_str.strip()
          non_terminal.add_production(BNF.Production(production_str, terminal_symbols, rng=rng))
    bnf_grammar = BNF.Grammar(list(terminal_symbols.values()), start_symbol)
    return bnf_grammar


@QuestionRegistry.register("LanguageQuestion")
class ValidStringsInLanguageQuestion(LanguageQuestion):
  MAX_TRIES = 1000
  
  def __init__(self, grammar_str_good: Optional[str] = None, grammar_str_bad: Optional[str] = None, *args, **kwargs):
    # Preserve question-specific params for QR code config
    if grammar_str_good is not None:
      kwargs['grammar_str_good'] = grammar_str_good
    if grammar_str_bad is not None:
      kwargs['grammar_str_bad'] = grammar_str_bad

    super().__init__(*args, **kwargs)
    self.answer_kind = ca.Answer.CanvasAnswerKind.MULTIPLE_ANSWER

  @classmethod
  def _select_random_grammar(cls, rng):
    """Select and return a random grammar configuration."""
    which_grammar = rng.choice(range(4))

    if which_grammar == 0:
      # todo: make a few different kinds of grammars that could be picked
      grammar_str_good = """
        <expression> ::= <term> | <expression> + <term> | <expression> - <term>
        <term>       ::= <factor> | <term> * <factor> | <term> / <factor>
        <factor>     ::= <number>
        <number>     ::= <digit> | <number> <digit>
        <digit>      ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
      """
      # Adding in a plus to number
      grammar_str_bad = """
        <expression> ::= <term> | <expression> + <term> | <expression> - <term>
        <term>       ::= <factor> | <term> * <factor> | <term> / <factor>
        <factor>     ::= <number>
        <number>     ::= <digit> + | <digit> <number>
        <digit>      ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
      """
      include_spaces = False
      max_length = 30
    elif which_grammar == 1:
      grammar_str_good = """
        <sentence> ::= <subject> <verb> <object>
        <subject> ::= The cat | A dog | The bird | A child | <adjective> <animal>
        <animal> ::= cat | dog | bird | child
        <adjective> ::= happy | sad | angry | playful
        <verb> ::= chases | sees | hates | loves
        <object> ::= the ball | the toy | the tree | <adjective> <object>
      """
      grammar_str_bad = """
        <sentence> ::= <subject> <verb> <object>
        <subject> ::= The human | The dog | A bird | Some child | A <adjective> <animal>
        <animal> ::= cat | dog | bird | child
        <adjective> ::= happy | sad | angry | playful
        <verb> ::= chases | sees | hates | loves
        <object> ::= the ball | the toy | the tree | <adjective> <object>
      """
      include_spaces = True
      max_length = 100
    elif which_grammar == 2:
      grammar_str_good = """
        <poem> ::= <line> | <line> <poem>
        <line> ::= <subject> <verb> <object> <modifier>
        <subject> ::= whispers | shadows | dreams | echoes | <compound-subject>
        <compound-subject> ::= <subject> and <subject>
        <verb> ::= dance | dissolve | shimmer | collapse | <compound-verb>
        <compound-verb> ::= <verb> then <verb>
        <object> ::= beneath | between | inside | around | <compound-object>
        <compound-object> ::= <object> through <object>
        <modifier> ::= silently | violently | mysteriously | endlessly | <recursive-modifier>
        <recursive-modifier> ::= <modifier> and <modifier>
      """
      grammar_str_bad = """
        <bad-poem> ::= <almost-valid-line> | <bad-poem> <bad-poem>
        <almost-valid-line> ::= <tricky-subject> <tricky-verb> <tricky-object> <tricky-modifier>
        <tricky-subject> ::= whispers | shadows and and | <duplicate-subject>
        <duplicate-subject> ::= whispers whispers
        <tricky-verb> ::= dance | <incorrect-verb-placement> | <verb-verb>
        <incorrect-verb-placement> ::= dance dance
        <verb-verb> ::= dance whispers
        <tricky-object> ::= beneath | <object-verb-swap> | <duplicate-object>
        <object-verb-swap> ::= dance beneath
        <duplicate-object> ::= beneath beneath
        <tricky-modifier> ::= silently | <modifier-subject-swap> | <duplicate-modifier>
        <modifier-subject-swap> ::= whispers silently
        <duplicate-modifier> ::= silently silently
      """
      include_spaces = True
      max_length = 100
    elif which_grammar == 3:
      grammar_str_good = """
        <A> ::= a <B> a |
        <B> ::= b <C> b |
        <C> ::= c <A> c |
      """
      grammar_str_bad = """
        <A> ::= a <B> c
        <B> ::= b <C> a |
        <C> ::= c <A> b |
      """
      include_spaces = False
      max_length = 100

    return {
      "grammar_str_good": grammar_str_good,
      "grammar_str_bad": grammar_str_bad,
      "include_spaces": include_spaces,
      "max_length": max_length,
    }
  
  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    rng = random.Random(rng_seed)

    grammar_str_good = kwargs.get("grammar_str_good")
    grammar_str_bad = kwargs.get("grammar_str_bad")

    if grammar_str_good is not None and grammar_str_bad is not None:
      include_spaces = kwargs.get("include_spaces", False)
      max_length = kwargs.get("max_length", 30)
    else:
      selection = cls._select_random_grammar(rng)
      grammar_str_good = selection["grammar_str_good"]
      grammar_str_bad = selection["grammar_str_bad"]
      include_spaces = selection["include_spaces"]
      max_length = selection["max_length"]

    grammar_good = BNF.parse_bnf(grammar_str_good, rng)
    grammar_bad = BNF.parse_bnf(grammar_str_bad, rng)

    num_answer_options = kwargs.get("num_answer_options", 4)
    num_answer_blanks = kwargs.get("num_answer_blanks", 4)

    answer_options = []

    good_string = grammar_good.generate(include_spaces)
    answer_options.append(ca.Answer(
      value=good_string,
      kind=ca.Answer.CanvasAnswerKind.MULTIPLE_ANSWER,
      correct=True
    ))

    bad_string = grammar_bad.generate(include_spaces)
    answer_options.append(ca.Answer(
      value=bad_string,
      kind=ca.Answer.CanvasAnswerKind.MULTIPLE_ANSWER,
      correct=False
    ))

    bad_early_string = grammar_bad.generate(include_spaces, early_exit=True)
    answer_options.append(ca.Answer(
      value=bad_early_string,
      kind=ca.Answer.CanvasAnswerKind.MULTIPLE_ANSWER,
      correct=False
    ))

    answer_text_set = {a.value for a in answer_options}
    num_tries = 0
    while len(answer_options) < 10 and num_tries < cls.MAX_TRIES:
      correct = rng.choice([True, False])
      if not correct:
        early_exit = rng.choice([True, False])
      else:
        early_exit = False

      generated_string = (
        grammar_good
        if correct or early_exit
        else grammar_bad
      ).generate(include_spaces, early_exit=early_exit)

      is_correct = correct and not early_exit

      if len(generated_string) < max_length and generated_string not in answer_text_set:
        answer_options.append(ca.Answer(
          value=generated_string,
          kind=ca.Answer.CanvasAnswerKind.MULTIPLE_ANSWER,
          correct=is_correct
        ))
        answer_text_set.add(generated_string)
      num_tries += 1

    featured_answers = {
      grammar_good.generate(),
      grammar_bad.generate(),
      grammar_good.generate(early_exit=True)
    }
    while len(featured_answers) < num_answer_options:
      featured_answers.add(
        rng.choice([
          lambda: grammar_good.generate(),
          lambda: grammar_bad.generate(),
          lambda: grammar_good.generate(early_exit=True),
        ])()
      )

    return {
      "grammar_good": grammar_good,
      "answer_options": answer_options,
      "featured_answers": featured_answers,
      "include_spaces": include_spaces,
      "num_answer_blanks": num_answer_blanks,
    }

  @classmethod
  def _build_body(cls, context):
    """Build question body and collect answers."""
    body = ca.Section()

    body.add_element(
      ca.OnlyHtml([
        ca.Paragraph([
          "Given the following grammar, which of the below strings are part of the language?"
        ])
      ])
    )
    body.add_element(
      ca.OnlyLatex([
        ca.Paragraph([
          "Given the following grammar "
          "please circle any provided strings that are part of the language (or indicate clearly if there are none), "
          "and on each blank line provide generate a new, unique string that is part of the language."
        ])
      ])
    )

    body.add_element(
      ca.Code(context["grammar_good"].get_grammar_string())
    )

    # Add in some answers as latex-only options to be circled
    latex_list = ca.OnlyLatex([])
    for answer in context["featured_answers"]:
      latex_list.add_element(ca.Paragraph([f"- `{str(answer)}`"]))
    body.add_element(latex_list)

    # For Latex-only, ask students to generate some more.
    body.add_element(
      ca.OnlyLatex([
        ca.AnswerBlock([ca.AnswerTypes.String("", label="") for i in range(context["num_answer_blanks"])])
      ])
    )

    return body, list(context["answer_options"])

  @classmethod
  def _build_explanation(cls, context):
    """Build question explanation."""
    explanation = ca.Section()
    explanation.add_element(
      ca.Paragraph([
        "Remember, for a string to be part of our language, we need to be able to derive it from our grammar.",
        "Unfortunately, there isn't space here to demonstrate the derivation so please work through them on your own!"
      ])
    )
    return explanation

  
