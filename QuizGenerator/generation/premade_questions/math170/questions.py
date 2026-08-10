"""MATH170 discrete-mathematics readiness questions."""

from QuizGenerator.generation.premade_questions.math_common import make_question


def _set_union(rng, **kwargs):
  return {"question": "Find the number of elements in the union of the two sets.", "equation": "|A|=7,\\quad |B|=5,\\quad |A\\cap B|=2", "answer": 10, "answer_kind": "int", "label": "|A union B|", "explanation": ["Add the sizes of A and B, but subtract the overlap because those elements were counted twice.", "There are 7 + 5 - 2 = 10 elements in the union."], "explanation_equations": ["|A\\cup B|=7+5-2=10"]}


def _logic(rng, **kwargs):
  return {"question": "If P is true and Q is false, evaluate the statement.", "equation": "P\\land\\neg Q", "answer": "TRUE", "answer_kind": "string", "label": "Truth value", "explanation": ["Q is false, so not Q is true. A conjunction is true only when both parts are true.", "Here P is true and not Q is true, so the whole statement is true."]}


def _contrapositive(rng, **kwargs):
  return {"question": "Write the contrapositive of: If an integer is divisible by 4, then it is even.", "equation": "4\\mid n \\Rightarrow n\\text{ is even}", "answer": "If n is not even, then 4 does not divide n.", "answer_kind": "string", "label": "Contrapositive", "explanation": ["To form a contrapositive, reverse the implication and negate both statements.", "The original conclusion becomes the new condition: if n is not even, then n cannot be divisible by 4."]}


def _sequence(rng, **kwargs):
  first, difference, n = rng.randint(1, 5), rng.randint(2, 7), rng.randint(5, 10)
  value = first + (n-1)*difference
  return {"question": "Find the requested term of this arithmetic sequence.", "equation": f"a_1={first},\\quad d={difference},\\quad a_{n}=?", "answer": value, "answer_kind": "int", "label": f"a_{n}", "explanation": [f"An arithmetic sequence adds {difference} each time. From the first term to term {n}, there are {n-1} additions.", f"Start at {first} and add {n-1} groups of {difference} to get {value}."], "explanation_equations": [f"a_{n}={first}+({n-1})({difference})={value}"]}


def _summation(rng, **kwargs):
  n = rng.randint(3, 8)
  value = n*(n+1)//2
  return {"question": "Evaluate the summation.", "equation": f"\\sum_{{k=1}}^{{{n}}} k", "answer": value, "answer_kind": "int", "label": "Sum", "explanation": ["This sum adds every integer from 1 through the upper index. The formula for this total is n(n+1)/2.", f"With n = {n}, the sum is {n}({n+1})/2 = {value}."], "explanation_equations": [f"\\frac{{{n}({n+1})}}{{2}}={value}"]}


def _bijection(rng, **kwargs):
  return {"question": "Is f(x) = x + 3 from the integers to the integers bijective? Enter YES or NO.", "equation": "f(x)=x+3", "answer": "YES", "answer_kind": "string", "label": "Bijective?", "explanation": ["Adding 3 never sends two different inputs to the same output, so the function is one-to-one. Every integer y has the preimage y - 3, so it is onto.", "Because it is both one-to-one and onto, the function is bijective."]}


def _equivalence(rng, **kwargs):
  return {"question": "For the relation a ~ b when a and b have the same parity, are 3 and 7 related? Enter YES or NO.", "equation": "a\\sim b \\iff a\\equiv b\\pmod 2", "answer": "YES", "answer_kind": "string", "label": "Related?", "explanation": ["Two numbers have the same parity when they leave the same remainder after division by 2. Both 3 and 7 are odd.", "They both have remainder 1 modulo 2, so 3 is related to 7."]}


def _counting(rng, **kwargs):
  a, b = rng.randint(3, 7), rng.randint(2, 5)
  return {"question": "How many outcomes are possible when choosing one item from each independent group?", "equation": f"{a}\\text{{ shirt choices and }}{b}\\text{{ pants choices}}", "answer": a*b, "answer_kind": "int", "label": "Outcomes", "explanation": [f"For each of the {a} shirt choices, there are {b} pants choices. The multiplication principle tells us to multiply independent choices.", f"That gives {a} times {b}, or {a*b} possible outfits."], "explanation_equations": [f"{a}\\cdot {b}={a*b}"]}


def _probability(rng, **kwargs):
  favorable, total = rng.randint(1, 5), rng.randint(6, 12)
  return {"question": "A fair experiment has the stated number of favorable outcomes and total outcomes. Find the probability.", "equation": f"\\text{{favorable}}={favorable},\\quad \\text{{total}}={total}", "answer": favorable/total, "label": "Probability", "explanation": ["For equally likely outcomes, probability is favorable outcomes divided by total outcomes.", f"There are {favorable} favorable outcomes out of {total}, so the probability is {favorable}/{total}."], "explanation_equations": [f"P=\\frac{{{favorable}}}{{{total}}}={favorable/total}"]}


def _graph_edges(rng, **kwargs):
  vertices = rng.randint(4, 8)
  return {"question": "How many edges does a complete graph with the stated number of vertices have?", "equation": f"K_{{{vertices}}}", "answer": vertices*(vertices-1)//2, "answer_kind": "int", "label": "Edges", "explanation": ["In a complete graph, every pair of different vertices has exactly one edge. The number of pairs is n(n-1)/2.", f"For {vertices} vertices, this is {vertices}({vertices-1})/2 = {vertices*(vertices-1)//2}."], "explanation_equations": [f"\\binom{{{vertices}}}{{2}}={vertices*(vertices-1)//2}"]}


for _name, _builder in {"SetUnion": _set_union, "LogicalStatement": _logic, "Contrapositive": _contrapositive, "ArithmeticSequence": _sequence, "Summation": _summation, "BijectiveFunction": _bijection, "EquivalenceRelation": _equivalence, "CountingPrinciple": _counting, "DiscreteProbability": _probability, "CompleteGraphEdges": _graph_edges}.items():
  globals()[_name] = make_question("math170", _name, _builder)
