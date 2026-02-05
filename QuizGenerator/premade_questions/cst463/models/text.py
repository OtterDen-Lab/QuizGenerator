import logging
from typing import List, Tuple

import numpy as np

import QuizGenerator.contentast as ca
from QuizGenerator.mixins import TableQuestionMixin
from QuizGenerator.premade_questions.cst463.models.matrices import MatrixQuestion
from QuizGenerator.question import QuestionRegistry

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
  
  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    rng = cls.get_rng(rng_seed)
    digits = cls.get_digits_to_round(digits_to_round=kwargs.get("digits_to_round"))

    embed_dim = kwargs.get("embed_dim", 3)
    num_contexts = kwargs.get("num_contexts", 3)

    vocab = [
      'cat', 'dog', 'run', 'jump', 'happy', 'sad', 'tree', 'house',
      'walk', 'sleep', 'fast', 'slow', 'big', 'small'
    ]

    selected_words = rng.choice(vocab, size=num_contexts + 1, replace=False)
    center_word = selected_words[0]
    context_words = selected_words[1:]

    center_emb = cls.get_rounded_matrix(rng, (embed_dim,), -2, 3, digits)
    context_embs = cls.get_rounded_matrix(rng, (num_contexts, embed_dim), -2, 3, digits)

    logits, probs = cls.skipgram_predict(center_emb, context_embs)

    most_likely_idx = np.argmax(probs)
    most_likely_word = context_words[most_likely_idx]

    return {
      "digits": digits,
      "embed_dim": embed_dim,
      "num_contexts": num_contexts,
      "center_word": center_word,
      "context_words": context_words,
      "center_emb": center_emb,
      "context_embs": context_embs,
      "logits": logits,
      "probs": probs,
      "most_likely_word": most_likely_word,
    }

  @classmethod
  def _build_body(cls, context) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question body and collect answers."""
    body = ca.Section()
    answers = []

    body.add_element(
      ca.Paragraph([
        f"Given center word: `{context['center_word']}` with embedding {context['center_emb']}, compute the skip-gram probabilities for each context word and identify the most likely one."
      ])
    )
    body.add_elements([
      ca.Paragraph([ca.Text(f"`{w}` : "), str(e)]) for w, e in zip(context["context_words"], context["context_embs"])
    ])

    logits_answer = ca.AnswerTypes.Vector(context["logits"], label="Logits")
    center_word_answer = ca.AnswerTypes.String(context["most_likely_word"], label="Most likely context word")
    answers.append(logits_answer)
    answers.append(center_word_answer)
    body.add_elements([
      ca.LineBreak(),
      logits_answer,
      ca.LineBreak(),
      center_word_answer
    ])

    log.debug(f"output: {context['logits']}")
    log.debug(f"weights: {context['probs']}")

    return body, answers

  @classmethod
  def _build_explanation(cls, context) -> Tuple[ca.Section, List[ca.Answer]]:
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
    center_emb_str = "[" + ", ".join([f"{x:.{digits}f}" for x in context["center_emb"]]) + "]"
    explanation.add_element(
      ca.Paragraph([
        f"Center word `{context['center_word']}`: {center_emb_str}"
      ])
    )

    explanation.add_element(
      ca.Paragraph([
        "Context words:"
      ])
    )

    for i, (word, emb) in enumerate(zip(context["context_words"], context["context_embs"])):
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
        f"Example: Logit for `{context['context_words'][0]}`"
      ])
    )

    context_emb = context["context_embs"][0]
    dot_product_terms = " + ".join([f"({context['center_emb'][j]:.2f} \\times {context_emb[j]:.2f})"
                                    for j in range(len(context["center_emb"]))])
    logit_val = context["logits"][0]

    explanation.add_element(
      ca.Equation(f"{dot_product_terms} = {logit_val:.2f}")
    )

    logits_str = "[" + ", ".join([f"{x:.2f}" for x in context["logits"]]) + "]"
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

    exp_logits = np.exp(context["logits"])
    sum_exp = exp_logits.sum()

    exp_terms = " + ".join([f"e^{{{l:.{digits}f}}}" for l in context["logits"]])

    explanation.add_element(
      ca.Equation(f"\\text{{denominator}} = {exp_terms} = {sum_exp:.{digits}f}")
    )

    explanation.add_element(
      ca.Paragraph([
        "Probabilities:"
      ])
    )

    for i, (word, prob) in enumerate(zip(context["context_words"], context["probs"])):
      explanation.add_element(
        ca.Equation(f"P(\\text{{{word}}}) = \\frac{{e^{{{context['logits'][i]:.{digits}f}}}}}{{{sum_exp:.{digits}f}}} = {prob:.{digits}f}")
      )

    # Step 4: Identify most likely
    most_likely_idx = np.argmax(context["probs"])
    most_likely_word = context["context_words"][most_likely_idx]

    explanation.add_element(
      ca.Paragraph([
        ca.Text("Conclusion:", emphasis=True),
        f" The most likely context word is `{most_likely_word}` with probability {context['probs'][most_likely_idx]:.{digits}f}"
      ])
    )

    return explanation, []
