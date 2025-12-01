import abc
import logging
import math
import keras
import numpy as np

from QuizGenerator.question import Question, QuestionRegistry, Answer
from QuizGenerator.contentast import ContentAST
from QuizGenerator.constants import MathRanges
from QuizGenerator.mixins import TableQuestionMixin

log = logging.getLogger(__name__)


@QuestionRegistry.register("cst463.word2vec.skipgram")
class word2vec__skipgram(Question, TableQuestionMixin):
  
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
    
    embed_dim = 3
    num_contexts = 3
    
    # Vocabulary pool
    vocab = ['cat', 'dog', 'run', 'jump', 'happy', 'sad', 'tree', 'house',
             'walk', 'sleep', 'fast', 'slow', 'big', 'small']
    
    # Sample words
    self.selected_words = self.rng.choice(vocab, size=num_contexts + 1, replace=False)
    self.center_word = self.selected_words[0]
    self.context_words = self.selected_words[1:]
    
    # Small integer embeddings
    self.center_emb = self.rng.randint(-2, 3, size=embed_dim)
    self.context_embs = self.rng.randint(-2, 3, size=(num_contexts, embed_dim))
    
    self.logits, self.probs = self.skipgram_predict(self.center_emb, self.context_embs)

    ## Answers:
    # center_word, center_emb, context_words, context_embs, logits, probs
    
    return True
  
  def get_body(self, **kwargs) -> ContentAST.Section:
    body = ContentAST.Section()
    
    body.add_element(
      ContentAST.Text(f"Given center word: `{self.center_word}` with embedding {self.center_emb}, , compute the skip-gram probabilities for each context word and identify the most likely one.")
    )
    body.add_elements([
      ContentAST.Paragraph([ContentAST.Text(f"`{w}` : "), ContentAST.Text(e)]) for w, e in zip(self.context_words, self.context_embs)
    ])
    
    
    log.debug(f"output: {self.logits}")
    log.debug(f"weights: {self.probs}")
    
    return body
  
  def get_explanation(self, **kwargs) -> ContentAST.Section:
    explanation = ContentAST.Section()
    
    return explanation

