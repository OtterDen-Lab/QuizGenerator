import abc
import logging
import math
import keras
import numpy as np
from typing import List, Tuple

from QuizGenerator.premade_questions.cst463.models.matrices import MatrixQuestion
from QuizGenerator.question import Question, QuestionRegistry
import QuizGenerator.contentast as ca
from QuizGenerator.constants import MathRanges
from QuizGenerator.mixins import TableQuestionMixin

log = logging.getLogger(__name__)


@QuestionRegistry.register("cst463.word2vec.skipgram")
class word2vec__skipgram(MatrixQuestion, TableQuestionMixin):
  
  @staticmethod
  def skipgram_predict(center_emb, context_embs):
    """
    center_emb: (embed_dim,) - center word embedding
    context_embs: (num_contexts, embed_dim) - context candidate embeddings

    Returns: probabilities (num_contexts,)
    """
    # Compute dot products (logits)
    logits = context_embs @ center_emb
    
    # Softmax
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum()
    
    return logits, probs
  
  def refresh(self, *args, **kwargs):
    super().refresh(*args, **kwargs)
    self.rng = np.random.RandomState(kwargs.get("rng_seed", None))
    
    embed_dim = kwargs.get("embed_dim", 3)
    num_contexts = kwargs.get("num_contexts", 3)
    
    # Vocabulary pool
    vocab = ['cat', 'dog', 'run', 'jump', 'happy', 'sad', 'tree', 'house',
             'walk', 'sleep', 'fast', 'slow', 'big', 'small']
    
    # Sample words
    self.selected_words = self.rng.choice(vocab, size=num_contexts + 1, replace=False)
    self.center_word = self.selected_words[0]
    self.context_words = self.selected_words[1:]
    
    # Small integer embeddings

    self.center_emb = self.get_rounded_matrix((embed_dim,), -2, 3)
    self.context_embs = self.get_rounded_matrix((num_contexts, embed_dim), -2, 3)
    
    self.logits, self.probs = self.skipgram_predict(self.center_emb, self.context_embs)

    ## Answers:
    # center_word, center_emb, context_words, context_embs, logits, probs
    self.answers["logits"] = ca.AnswerTypes.Vector(self.logits, label="Logits")
    most_likely_idx = np.argmax(self.probs)
    most_likely_word = self.context_words[most_likely_idx]
    self.answers["center_word"] = ca.AnswerTypes.String(most_likely_word, label="Most likely context word")
    
    
    return True
  
  def _get_body(self, **kwargs) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question body and collect answers."""
    body = ca.Section()
    answers = []

    body.add_element(
      ca.Paragraph([
        f"Given center word: `{self.center_word}` with embedding {self.center_emb}, compute the skip-gram probabilities for each context word and identify the most likely one."
      ])
    )
    body.add_elements([
      ca.Paragraph([ca.Text(f"`{w}` : "), str(e)]) for w, e in zip(self.context_words, self.context_embs)
    ])

    answers.append(self.answers["logits"])
    answers.append(self.answers["center_word"])
    body.add_elements([
      ca.LineBreak(),
      self.answers["logits"],
      ca.LineBreak(),
      self.answers["center_word"]
    ])

    log.debug(f"output: {self.logits}")
    log.debug(f"weights: {self.probs}")

    return body, answers

  def get_body(self, **kwargs) -> ca.Section:
    """Build question body (backward compatible interface)."""
    body, _ = self._get_body(**kwargs)
    return body
  
  def _get_explanation(self, **kwargs) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question explanation."""
    explanation = ca.Section()
    digits = ca.Answer.DEFAULT_ROUNDING_DIGITS

    explanation.add_element(
      ca.Paragraph([
        "In the skip-gram model, we predict context words given a center word by computing dot products between embeddings and applying softmax."
      ])
    )

    # Step 1: Show embeddings
    explanation.add_element(
      ca.Paragraph([
        ca.Text("Step 1: Given embeddings", emphasis=True)
      ])
    )

    # Format center embedding
    center_emb_str = "[" + ", ".join([f"{x:.{digits}f}" for x in self.center_emb]) + "]"
    explanation.add_element(
      ca.Paragraph([
        f"Center word `{self.center_word}`: {center_emb_str}"
      ])
    )

    explanation.add_element(
      ca.Paragraph([
        "Context words:"
      ])
    )

    for i, (word, emb) in enumerate(zip(self.context_words, self.context_embs)):
      emb_str = "[" + ", ".join([f"{x:.2f}" for x in emb]) + "]"
      explanation.add_element(
        ca.Paragraph([
          f"`{word}`: {emb_str}"
        ])
      )

    # Step 2: Compute logits (dot products)
    explanation.add_element(
      ca.Paragraph([
        ca.Text("Step 2: Compute logits (dot products)", emphasis=True)
      ])
    )

    # Show ONE example
    explanation.add_element(
      ca.Paragraph([
        f"Example: Logit for `{self.context_words[0]}`"
      ])
    )

    context_emb = self.context_embs[0]
    dot_product_terms = " + ".join([f"({self.center_emb[j]:.2f} \\times {context_emb[j]:.2f})"
                                    for j in range(len(self.center_emb))])
    logit_val = self.logits[0]

    explanation.add_element(
      ca.Equation(f"{dot_product_terms} = {logit_val:.2f}")
    )

    logits_str = "[" + ", ".join([f"{x:.2f}" for x in self.logits]) + "]"
    explanation.add_element(
      ca.Paragraph([
        f"All logits: {logits_str}"
      ])
    )

    # Step 3: Apply softmax
    explanation.add_element(
      ca.Paragraph([
        ca.Text("Step 3: Apply softmax to get probabilities", emphasis=True)
      ])
    )

    exp_logits = np.exp(self.logits)
    sum_exp = exp_logits.sum()

    exp_terms = " + ".join([f"e^{{{l:.{digits}f}}}" for l in self.logits])

    explanation.add_element(
      ca.Equation(f"\\text{{denominator}} = {exp_terms} = {sum_exp:.{digits}f}")
    )

    explanation.add_element(
      ca.Paragraph([
        "Probabilities:"
      ])
    )

    for i, (word, prob) in enumerate(zip(self.context_words, self.probs)):
      explanation.add_element(
        ca.Equation(f"P(\\text{{{word}}}) = \\frac{{e^{{{self.logits[i]:.{digits}f}}}}}{{{sum_exp:.{digits}f}}} = {prob:.{digits}f}")
      )

    # Step 4: Identify most likely
    most_likely_idx = np.argmax(self.probs)
    most_likely_word = self.context_words[most_likely_idx]

    explanation.add_element(
      ca.Paragraph([
        ca.Text("Conclusion:", emphasis=True),
        f" The most likely context word is `{most_likely_word}` with probability {self.probs[most_likely_idx]:.{digits}f}"
      ])
    )

    return explanation, []

  def get_explanation(self, **kwargs) -> ca.Section:
    """Build question explanation (backward compatible interface)."""
    explanation, _ = self._get_explanation(**kwargs)
    return explanation

