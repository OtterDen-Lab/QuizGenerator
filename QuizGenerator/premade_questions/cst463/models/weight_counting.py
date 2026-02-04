from __future__ import annotations

import abc
import logging
import random
try:
  import keras
except ImportError as exc:
  keras = None
  _KERAS_IMPORT_ERROR = exc
else:
  _KERAS_IMPORT_ERROR = None
import numpy as np
from typing import List, Tuple

from QuizGenerator.question import Question, QuestionRegistry
import QuizGenerator.contentast as ca

log = logging.getLogger(__name__)


class WeightCounting(Question, abc.ABC):
  @staticmethod
  def _ensure_keras():
    if keras is None:
      raise ImportError(
        "Keras is required for CST463 model questions. "
        "Install with: pip install 'QuizGenerator[cst463]'"
      ) from _KERAS_IMPORT_ERROR

  @staticmethod
  @abc.abstractmethod
  def get_model(rng: random.Random) -> keras.Model:
    pass
  
  @staticmethod
  def model_to_python(model: keras.Model, fields=None, include_input=True):
    WeightCounting._ensure_keras()
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
  
  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    rng = random.Random(rng_seed)

    refresh_success = False
    while not refresh_success:
      try:
        model, fields = cls.get_model(rng)
        refresh_success = True
      except ValueError as e:
        log.error(e)
        log.info(f"Regenerating {cls.__name__} due to improper configuration")
        continue

    num_parameters = model.count_params()

    return {
      "model": model,
      "fields": fields,
      "num_parameters": num_parameters,
    }

  @classmethod
  def _build_body(cls, context) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question body and collect answers."""
    body = ca.Section()
    answers = []

    body.add_element(
      ca.Paragraph(
        [
          ca.Text("Given the below model, how many parameters does it use?")
        ]
      )
    )

    body.add_element(
      ca.Code(
        cls.model_to_python(
          context["model"],
          fields=context["fields"]
        )
      )
    )

    body.add_element(ca.LineBreak())

    num_parameters_answer = ca.AnswerTypes.Int(context["num_parameters"], label="Number of Parameters")
    answers.append(num_parameters_answer)
    body.add_element(num_parameters_answer)

    return body, answers

  @classmethod
  def _build_explanation(cls, context) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question explanation."""
    explanation = ca.Section()

    def markdown_summary(model) -> ca.Table:
      # Ensure the model is built by running build() or calling it once
      if not model.built:
        try:
          model.build(model.input_shape)
        except:
          pass  # Some subclassed models need real data to build

      data = []

      total_params = 0

      for layer in model.layers:
        name = layer.name
        ltype = layer.__class__.__name__

        # Try to extract output shape
        try:
          outshape = tuple(layer.output.shape)
        except:
          outshape = "?"

        params = layer.count_params()
        total_params += params

        data.append([name, ltype, outshape, params])

      data.append(["**Total**", "", "", f"**{total_params}**"])
      return ca.Table(data=data, headers=["Layer", "Type", "Output Shape", "Params"])


    summary_lines = []
    context["model"].summary(print_fn=lambda x: summary_lines.append(x))
    explanation.add_element(
      # ca.Text('\n'.join(summary_lines))
      markdown_summary(context["model"])
    )

    return explanation, []


@QuestionRegistry.register("cst463.WeightCounting-CNN")
class WeightCounting_CNN(WeightCounting):
  
  @staticmethod
  def get_model(rng: random.Random) -> tuple[keras.Model, list[str]]:
    WeightCounting._ensure_keras()
    input_size = rng.choice(np.arange(28, 32))
    cnn_num_filters = rng.choice(2 ** np.arange(8))
    cnn_kernel_size = rng.choice(1 + np.arange(10))
    cnn_strides = rng.choice(1 + np.arange(10))
    pool_size = rng.choice(1 + np.arange(10))
    pool_strides = rng.choice(1 + np.arange(10))
    num_output_size = rng.choice([1, 10, 32, 100])
    
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
    return model, ["units", "filters", "kernel_size", "strides", "padding", "pool_size"]


@QuestionRegistry.register("cst463.WeightCounting-RNN")
class WeightCounting_RNN(WeightCounting):
  @staticmethod
  def get_model(rng: random.Random) -> tuple[keras.Model, list[str]]:
    WeightCounting._ensure_keras()
    timesteps = int(rng.choice(np.arange(20, 41)))
    feature_size = int(rng.choice(np.arange(8, 65)))

    rnn_units = int(rng.choice(2 ** np.arange(4, 9)))
    rnn_type = rng.choice(["SimpleRNN"])
    return_sequences = bool(rng.choice([True, False]))

    num_output_size = int(rng.choice([1, 10, 32, 100]))

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


# ConvolutionCalculation is implemented in cnns.py
