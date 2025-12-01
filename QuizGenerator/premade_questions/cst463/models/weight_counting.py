import abc
import logging
import math
import keras
import numpy as np

from QuizGenerator.question import Question, QuestionRegistry, Answer
from QuizGenerator.contentast import ContentAST
from QuizGenerator.constants import MathRanges

log = logging.getLogger(__name__)


class WeightCounting(Question, abc.ABC):
  @abc.abstractmethod
  def get_model(self) -> keras.Model:
    pass
  
  @staticmethod
  def model_to_python(model: keras.Model, fields=None, include_input=True):
    if fields is None:
      fields = []
    
    def sanitize(v):
      """Convert numpy types to pure Python."""
      if isinstance(v, np.generic):  # np.int64, np.float32, etc.
        return v.item()
      if isinstance(v, (list, tuple)):
        return type(v)(sanitize(x) for x in v)
      if isinstance(v, dict):
        return {k: sanitize(x) for k, x in v.items()}
      return v
    
    lines = []
    lines.append("keras.models.Sequential([")
    
    # ---- Emit an Input line if we can ----
    # model.input_shape is like (None, H, W, C) or (None, D)
    if include_input and getattr(model, "input_shape", None) is not None:
      input_shape = sanitize(model.input_shape[1:])  # drop batch dimension
      # If it's a 1D shape like (784,), keep as tuple; if scalar, still fine.
      lines.append(f"  keras.layers.Input(shape={input_shape!r}),")
    
    # ---- Emit all other layers ----
    for layer in model.layers:
      # If user explicitly had an Input layer, we don't want to duplicate it
      if isinstance(layer, keras.layers.InputLayer):
        # You *could* handle it specially here, but usually we just skip
        continue
      
      cfg = layer.get_config()
      
      # If fields is empty, include everything; otherwise filter by fields.
      if fields:
        items = [(k, v) for k, v in cfg.items() if k in fields]
      else:
        items = cfg.items()
      
      arg_lines = [
        f"{k}={sanitize(v)!r}"  # !r so strings get quotes, etc.
        for k, v in items
      ]
      args = ",\n    ".join(arg_lines)
      
      lines.append(
        f"  keras.layers.{layer.__class__.__name__}("
        f"{'\n    ' if args else ''}{args}{'\n  ' if args else ''}),"
      )
    
    lines.append("])")
    return "\n".join(lines)
  
  def refresh(self, *args, **kwargs):
    super().refresh(*args, **kwargs)
    
    refresh_success = False
    while not refresh_success:
      try:
        self.model, self.fields = self.get_model()
        refresh_success = True
      except ValueError as e:
        log.error(e)
        log.info(f"Regenerating {self.__class__.__name__} due to improper configuration")
        continue
    
    self.num_parameters = self.model.count_params()
    self.answers["num_parameters"] = Answer.integer(
      "num_parameters",
      self.num_parameters
    )
    
    return True
  
  def get_body(self, **kwargs) -> ContentAST.Section:
    body = ContentAST.Section()
    
    body.add_element(
      ContentAST.Paragraph(
        [
          ContentAST.Text("Given the below model, how many parameters does it use?")
        ]
      )
    )
    
    body.add_elements(
      [
        ContentAST.Code(
          self.model_to_python(
            self.model,
            fields=self.fields
          )
        )
      ]
    )
    
    return body
  
  def get_explanation(self, **kwargs) -> ContentAST.Section:
    explanation = ContentAST.Section()
    
    explanation.add_element(
      ContentAST.Text(self.model.summary())
    )
    
    return explanation


@QuestionRegistry.register("cst463.WeightCounting-CNN")
class WeightCounting_CNN(WeightCounting):
  
  def get_model(self) -> tuple[keras.Model, list[str]]:
    input_size = self.rng.choice(np.arange(28, 32))
    cnn_num_filters = self.rng.choice(2 ** np.arange(8))
    cnn_kernel_size = self.rng.choice(1 + np.arange(10))
    cnn_strides = self.rng.choice(1 + np.arange(10))
    pool_size = self.rng.choice(1 + np.arange(10))
    pool_strides = self.rng.choice(1 + np.arange(10))
    num_output_size = self.rng.choice([1, 10, 32, 100])
    
    # Let's just make a small model
    model = keras.models.Sequential(
      [
        keras.layers.Input((input_size, input_size, 1)),
        keras.layers.Conv2D(
          filters=cnn_num_filters,
          kernel_size=(cnn_kernel_size, cnn_kernel_size),
          strides=(cnn_strides, cnn_strides),
          padding="valid"
        ),
        keras.layers.MaxPool2D(
          pool_size=(pool_size, pool_size),
          strides=(pool_strides, pool_strides)
        ),
        keras.layers.Dense(
          num_output_size
        )
      ]
    )
    return model, ["filters", "kernel_size", "strides", "padding", "pool_size"]


@QuestionRegistry.register("cst463.WeightCounting-RNN")
class WeightCounting_RNN(WeightCounting):
  def get_model(self) -> tuple[keras.Model, list[str]]:
    timesteps = int(self.rng.choice(np.arange(20, 41)))
    feature_size = int(self.rng.choice(np.arange(8, 65)))

    rnn_units = int(self.rng.choice(2 ** np.arange(4, 9)))
    rnn_type = self.rng.choice(["SimpleRNN"])
    return_sequences = bool(self.rng.choice([True, False]))

    num_output_size = int(self.rng.choice([1, 10, 32, 100]))

    RNNLayer = getattr(keras.layers, rnn_type)

    model = keras.models.Sequential([
      keras.layers.Input((timesteps, feature_size)),
      RNNLayer(
        units=rnn_units,
        return_sequences=return_sequences,
      ),
      keras.layers.Dense(num_output_size),
    ])
    return model, ["units", "return_sequences"]


@QuestionRegistry.register()
class ConvolutionCalculation(Question):
  pass
