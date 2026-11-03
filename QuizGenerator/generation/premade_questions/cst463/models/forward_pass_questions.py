"""Small, hand-calculable forward-pass questions for CNNs and sequences."""

from __future__ import annotations

from io import BytesIO

import numpy as np

import QuizGenerator.generation.contentast as ca
from QuizGenerator.generation.mixins import TableQuestionMixin
from QuizGenerator.generation.question import Question, QuestionRegistry


def _number(value, digits=4):
  """Format an integer or compact decimal with at most the requested places."""
  if abs(value - round(value)) < 1e-10:
    return str(int(round(value)))
  return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _inline_vector(values, digits=4):
  """Format a vector compactly without relying on matrix rendering."""
  return "(" + ", ".join(_number(value, digits) for value in values) + ")"


def _inline_matrix(values):
  """Format a small matrix compactly for Canvas and PDF question bodies."""
  return "[" + "; ".join(_inline_vector(row) for row in values) + "]"


def _add_labeled_answer(section, label, answer):
  """Add a math-rendered label followed by an answer blank."""
  section.add_element(ca.Paragraph([
    ca.Equation(label, inline=True),
    ": ",
    answer,
  ]))


class SequenceForwardPassQuestion(Question, TableQuestionMixin):
  """Shared defaults for current CST463 CNN and sequence exercises."""

  def __init__(self, *args, **kwargs):
    kwargs["topic"] = kwargs.get("topic", Question.Topic.ML_OPTIMIZATION)
    super().__init__(*args, **kwargs)


@QuestionRegistry.register("cst463.cnn.1d-forward-pass")
class Convolution1DForwardPass(SequenceForwardPassQuestion):
  """Compute a one-dimensional multi-channel cross-correlation feature map."""

  DEFAULT_INPUT_LENGTH = 6
  DEFAULT_KERNEL_SIZE = 3
  DEFAULT_INPUT_CHANNELS = 1

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    context = super()._build_context(rng_seed=rng_seed, **kwargs)
    input_length = kwargs.get("input_length", cls.DEFAULT_INPUT_LENGTH)
    kernel_size = kwargs.get("kernel_size", cls.DEFAULT_KERNEL_SIZE)
    input_channels = kwargs.get("input_channels", cls.DEFAULT_INPUT_CHANNELS)
    stride = kwargs.get("stride", 1)
    padding = kwargs.get("padding", 0)
    binary_filters = kwargs.get("binary_filters", True)
    include_bias = kwargs.get("include_bias", False)

    if input_length < 2 or kernel_size < 1:
      raise ValueError("input_length and kernel_size must be positive")
    if not 1 <= input_channels <= 3:
      raise ValueError("input_channels must be between 1 and 3")
    if stride < 1 or padding < 0:
      raise ValueError("stride must be positive and padding must be nonnegative")
    padded_length = input_length + 2 * padding
    if kernel_size > padded_length:
      raise ValueError("kernel_size cannot exceed the padded input length")

    inputs = np.array([
      [context.rng.randint(-100, 100) / 10 for _ in range(input_length)]
      for _ in range(input_channels)
    ], dtype=float)
    filter_values = (-1, 0, 1) if binary_filters else (-2, -1, 0, 1, 2)
    kernel = np.array([
      [context.rng.choice(filter_values) for _ in range(kernel_size)]
      for _ in range(input_channels)
    ], dtype=int)
    bias = context.rng.choice((-1, 0, 1)) if include_bias else 0
    padded_inputs = np.pad(inputs, ((0, 0), (padding, padding)))
    output_length = (padded_length - kernel_size) // stride + 1
    output = np.array([
      sum(
        int(np.dot(kernel[channel], padded_inputs[channel, position * stride:position * stride + kernel_size]))
        for channel in range(input_channels)
      ) + bias
      for position in range(output_length)
    ], dtype=float)

    context["input_length"] = input_length
    context["kernel_size"] = kernel_size
    context["input_channels"] = input_channels
    context["stride"] = stride
    context["padding"] = padding
    context["inputs"] = inputs
    context["kernel"] = kernel
    context["bias"] = bias
    context["include_bias"] = include_bias
    context["output"] = output
    return context

  @classmethod
  def is_interesting_ctx(cls, context) -> bool:
    return (
      np.unique(context.inputs).size >= min(4, context.inputs.size)
      and np.max(np.abs(context.inputs)) >= 2
      and np.any(context.output != 0)
      and (
        context.output.size == 1 or len(set(context.output.tolist())) > 1
      )
    )

  @classmethod
  def _build_body(cls, context):
    body = ca.Section()
    body.add_element(ca.Paragraph([
      "Compute the output of this one-dimensional convolution. Use "
      "cross-correlation as in the slides, so do not flip the kernel."
    ]))

    if context.input_channels == 1:
      body.add_element(ca.Paragraph([
        "Input sequence: ",
        _inline_vector(context.inputs[0], digits=1),
        ". Kernel: ",
        _inline_vector(context.kernel[0]),
        ".",
      ]))
    else:
      body.add_element(ca.Paragraph([
        "For each output position, multiply each kernel slice with its "
        "matching input channel, then add the channel contributions."
      ]))
      for channel in range(context.input_channels):
        body.add_element(ca.Paragraph([
          f"Input channel {channel + 1}: ",
          _inline_vector(context.inputs[channel], digits=1),
          ". Kernel slice ",
          f"{channel + 1}: ",
          _inline_vector(context.kernel[channel]),
          ".",
        ]))
    body.add_element(ca.Paragraph([
      f"Stride: {context.stride}; padding: {context.padding}; "
      f"bias: {context.bias}."
    ]))

    answer = ca.AnswerTypes.CompactVector(
      list(context.output),
      label="Output feature map",
    )
    body.add_element(ca.Paragraph([
      "Enter the output values from left to right as one list.",
    ]))
    body.add_element(ca.AnswerBlock(answer))
    return body, [answer]

  @classmethod
  def _build_explanation(cls, context):
    explanation = ca.Section()
    explanation.add_element(ca.Paragraph([
      "The filter has ",
      str(context.kernel_size),
      " entries, so each output value uses a consecutive window of ",
      str(context.kernel_size),
      " input values. For ",
      ca.Equation("y_0", inline=True),
      ", that window begins at the first input value. Each later output "
      "shifts the window by the stride, then repeats the same "
      "multiply-and-add calculation."
    ]))
    padded_inputs = np.pad(
      context.inputs,
      ((0, 0), (context.padding, context.padding)),
    )
    for position, result in enumerate(context.output):
      start = position * context.stride
      channel_terms = []
      for channel in range(context.input_channels):
        products = [
          f"({_number(context.kernel[channel, offset])}) \\cdot "
          f"({_number(padded_inputs[channel, start + offset])})"
          for offset in range(context.kernel_size)
        ]
        channel_terms.append(" + ".join(products))
      terms = " + ".join(f"({term})" for term in channel_terms)
      if context.include_bias:
        terms += f" + {_number(context.bias)}"
      window_end = start + context.kernel_size - 1
      if context.input_channels == 1:
        window = padded_inputs[0, start:start + context.kernel_size]
        explanation.add_element(ca.Paragraph([
          ca.Equation(f"y_{position}", inline=True),
          f" uses input positions {start} through {window_end}, which are ",
          ca.Text(_inline_vector(window, digits=1), emphasis=True),
          ".",
        ]))
      else:
        explanation.add_element(ca.Paragraph([
          ca.Equation(f"y_{position}", inline=True),
          f" uses positions {start} through {window_end} from each input channel.",
        ]))
      explanation.add_element(ca.Equation(
        f"y_{position} = {terms} = {_number(result)}",
        inline=False,
      ))
    explanation.add_element(ca.Paragraph([
      "Therefore the feature map is ",
      ca.Equation(
        "[" + ", ".join(_number(value) for value in context.output) + "]",
        inline=True,
      ),
      ".",
    ]))
    return explanation, []


@QuestionRegistry.register("cst463.cnn.max-pooling-forward-pass")
class MaxPoolingForwardPass(SequenceForwardPassQuestion):
  """Compute a two-dimensional max-pooling feature map."""

  DEFAULT_INPUT_SIZE = 4
  DEFAULT_POOL_SIZE = 2
  DEFAULT_STRIDE = 2

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    context = super()._build_context(rng_seed=rng_seed, **kwargs)
    input_height = kwargs.get("input_height", kwargs.get("input_size", cls.DEFAULT_INPUT_SIZE))
    input_width = kwargs.get("input_width", kwargs.get("input_size", cls.DEFAULT_INPUT_SIZE))
    pool_size = kwargs.get("pool_size", cls.DEFAULT_POOL_SIZE)
    stride = kwargs.get("stride", cls.DEFAULT_STRIDE)
    if min(input_height, input_width, pool_size, stride) < 1:
      raise ValueError("pooling dimensions and stride must be positive")
    if pool_size > min(input_height, input_width):
      raise ValueError("pool_size cannot exceed an input dimension")
    output_height = (input_height - pool_size) // stride + 1
    output_width = (input_width - pool_size) // stride + 1
    feature_map = np.array([
      [context.rng.randint(0, 9) for _ in range(input_width)]
      for _ in range(input_height)
    ], dtype=int)
    output = np.array([
      [
        np.max(feature_map[row * stride:row * stride + pool_size,
                           column * stride:column * stride + pool_size])
        for column in range(output_width)
      ]
      for row in range(output_height)
    ], dtype=int)
    context["feature_map"] = feature_map
    context["pool_size"] = pool_size
    context["stride"] = stride
    context["output"] = output
    return context

  @classmethod
  def is_interesting_ctx(cls, context) -> bool:
    return (
      context.output.size == 1
      or len(set(context.output.flatten().tolist())) > 1
    )

  @classmethod
  def _build_body(cls, context):
    body = ca.Section()
    body.add_element(ca.Paragraph([
      f"Apply {context.pool_size}×{context.pool_size} max pooling with "
      f"stride {context.stride} to this feature map."
    ]))
    body.add_element(ca.Matrix(data=context.feature_map.tolist(), bracket_type="b"))
    body.add_element(ca.Paragraph([
      "The pooled feature map has these entries:",
    ]))
    placeholder_matrix = [
      [rf"p_{{{row},{column}}}" for column in range(context.output.shape[1])]
      for row in range(context.output.shape[0])
    ]
    body.add_element(ca.Matrix(data=placeholder_matrix, bracket_type="b"))
    answers = [
      ca.AnswerTypes.Int(
        int(context.output[row, column]),
        label="",
      )
      for row in range(context.output.shape[0])
      for column in range(context.output.shape[1])
    ]
    for row in range(context.output.shape[0]):
      for column in range(context.output.shape[1]):
        answer_index = row * context.output.shape[1] + column
        _add_labeled_answer(
          body,
          f"p_{{{row},{column}}}",
          answers[answer_index],
        )
    return body, answers

  @classmethod
  def _build_explanation(cls, context):
    explanation = ca.Section()
    explanation.add_element(ca.Paragraph([
      f"A {context.pool_size}×{context.pool_size} pooling window keeps its "
      "largest entry. The stride says how far the window moves before "
      "computing the next output entry: with stride ",
      str(context.stride),
      ", it moves ",
      str(context.stride),
      " cells right for the next column and ",
      str(context.stride),
      " cells down for the next row."
    ]))
    for row_index in range(context.output.shape[0]):
      for column_index in range(context.output.shape[1]):
        start_row = row_index * context.stride
        start_column = column_index * context.stride
        window = context.feature_map[
          start_row:start_row + context.pool_size,
          start_column:start_column + context.pool_size,
        ]
        explanation.add_element(ca.MathExpression([
          f"p_{{{row_index},{column_index}}} = max(",
          ca.Matrix(data=window, bracket_type="b", inline=True),
          f") = {int(context.output[row_index, column_index])}",
        ]))
    explanation.add_element(ca.Paragraph(["The pooled feature map is:"]))
    explanation.add_element(ca.Matrix(data=context.output.tolist(), bracket_type="b"))
    return explanation, []


@QuestionRegistry.register("cst463.rnn.state-sequence-forward-pass")
class RNNStateSequenceForwardPass(SequenceForwardPassQuestion):
  """Compute every hidden state of a tanh RNN using the course notation."""

  DEFAULT_SEQUENCE_LENGTH = 3
  DEFAULT_INPUT_DIM = 1
  DEFAULT_HIDDEN_DIM = 2
  ROUND_DIGITS = 4

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    context = super()._build_context(rng_seed=rng_seed, **kwargs)
    sequence_length = kwargs.get("sequence_length", cls.DEFAULT_SEQUENCE_LENGTH)
    input_dim = kwargs.get("input_dim", cls.DEFAULT_INPUT_DIM)
    hidden_dim = kwargs.get("hidden_dim", cls.DEFAULT_HIDDEN_DIM)
    include_bias = kwargs.get("include_bias", True)
    if min(sequence_length, input_dim, hidden_dim) < 1:
      raise ValueError("RNN sequence_length and dimensions must be positive")

    values = (-1, 0, 1)
    inputs = np.array([
      [context.rng.choice(values) for _ in range(input_dim)]
      for _ in range(sequence_length)
    ], dtype=float)
    W_xh = np.array([
      [context.rng.choice(values) for _ in range(input_dim)]
      for _ in range(hidden_dim)
    ], dtype=float)
    W_hh = np.array([
      [context.rng.choice(values) for _ in range(hidden_dim)]
      for _ in range(hidden_dim)
    ], dtype=float)
    bias = np.array([
      context.rng.choice(values) if include_bias else 0
      for _ in range(hidden_dim)
    ], dtype=float)
    states = []
    traces = []
    h_previous = np.zeros(hidden_dim)
    for input_vector in inputs:
      input_contribution = W_xh @ input_vector
      hidden_contribution = W_hh @ h_previous
      preactivation = input_contribution + hidden_contribution + bias
      h_current = np.tanh(preactivation)
      traces.append({
        "input": input_vector.copy(),
        "previous": h_previous.copy(),
        "input_contribution": input_contribution.copy(),
        "hidden_contribution": hidden_contribution.copy(),
        "preactivation": preactivation.copy(),
        "state": h_current.copy(),
      })
      states.append(h_current.copy())
      h_previous = h_current

    context["sequence_length"] = sequence_length
    context["input_dim"] = input_dim
    context["hidden_dim"] = hidden_dim
    context["inputs"] = inputs
    context["W_xh"] = W_xh
    context["W_hh"] = W_hh
    context["bias"] = bias
    context["include_bias"] = include_bias
    context["states"] = np.array(states)
    context["traces"] = traces
    return context

  @classmethod
  def is_interesting_ctx(cls, context) -> bool:
    return (
      np.any(np.abs(context.states) > 0.05)
      and (
        context.states.shape[0] == 1
        or len({tuple(np.round(state, 4)) for state in context.states}) > 1
      )
    )

  @staticmethod
  def _generate_rnn_diagram() -> BytesIO:
    """Draw one recurrent cell instead of an unrolled time sequence."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5.5, 2.65))
    axis.axis("off")
    axis.text(
      2.7, 1.2, "RNN cell\n(tanh activation)",
      ha="center", va="center", fontsize=11,
      bbox={"boxstyle": "round,pad=0.6", "fc": "#d8eaf7", "ec": "#3478a6", "lw": 1.7},
    )
    axis.annotate(
      "", xy=(2.05, 0.85), xytext=(0.7, 0.35),
      arrowprops={"arrowstyle": "->", "color": "#3478a6", "lw": 1.7},
    )
    axis.text(0.48, 0.22, r"$x_t$", fontsize=14, ha="center", va="center")
    axis.annotate(
      "", xy=(2.05, 1.52), xytext=(0.7, 1.92),
      arrowprops={"arrowstyle": "->", "color": "#9a5b18", "lw": 1.7},
    )
    axis.text(0.42, 2.03, r"$h_{t-1}$", fontsize=14, ha="center", va="center")
    axis.annotate(
      "", xy=(4.8, 1.2), xytext=(3.42, 1.2),
      arrowprops={"arrowstyle": "->", "color": "#3478a6", "lw": 1.7},
    )
    axis.text(5.05, 1.2, r"$h_t$", fontsize=14, ha="center", va="center")
    axis.text(0.42, 2.42, "previous state", fontsize=9, ha="center", color="#7c4610")
    axis.text(0.48, -0.1, "current input", fontsize=9, ha="center", color="#245b80")
    axis.text(5.05, 0.92, "new state", fontsize=9, ha="center", color="#245b80")
    axis.set_xlim(0, 5.5)
    axis.set_ylim(-0.35, 2.6)
    figure.tight_layout(pad=0.15)
    image = BytesIO()
    figure.savefig(image, format="png", dpi=180, bbox_inches="tight",
                   facecolor="white", edgecolor="none")
    plt.close(figure)
    image.seek(0)
    return image

  @classmethod
  def _build_body(cls, context):
    body = ca.Section()
    body.add_element(ca.Paragraph([
      "Given the RNN cell below with a tanh activation function, calculate "
      "the hidden states ",
      ca.Equation("h_1", inline=True),
      ", ",
      ca.Equation("h_2", inline=True),
      ", and ",
      ca.Equation("h_3", inline=True),
      ". Use full calculator precision until rounding each submitted "
      "component to four decimal places."
    ]))
    body.add_element(ca.Picture(
      img_data=cls._generate_rnn_diagram(),
      width="100%",
    ))
    if context.input_dim == 1:
      body.add_element(ca.Paragraph([
        "Input sequence: ",
        ", ".join(_number(vector[0]) for vector in context.inputs),
        ".",
      ]))
    else:
      sequence_text = "; ".join(
        f"x_{timestep} = {_inline_vector(vector)}"
        for timestep, vector in enumerate(context.inputs, start=1)
      )
      body.add_element(ca.Paragraph(["Input sequence: ", sequence_text, "."]))
    body.add_element(ca.Paragraph([
      ca.Equation("W_{xh}", inline=True),
      " is:",
    ]))
    body.add_element(ca.Matrix(data=context.W_xh, bracket_type="b"))
    body.add_element(ca.Paragraph([
      ca.Equation("W_{hh}", inline=True),
      " is:",
    ]))
    body.add_element(ca.Matrix(data=context.W_hh, bracket_type="b"))
    body.add_element(ca.Paragraph([
      ca.Equation("b", inline=True),
      " is:",
    ]))
    body.add_element(ca.Matrix(
      data=[[value] for value in context.bias],
      bracket_type="b",
    ))
    body.add_element(ca.Paragraph([
      "The initial hidden state is ",
      ca.Equation("h_0", inline=True),
      " = ",
      _inline_vector([0] * context.hidden_dim),
      ".",
    ]))

    answers = [
      ca.AnswerTypes.Vector(
        list(state),
        label="",
      )
      for timestep, state in enumerate(context.states, start=1)
    ]
    for timestep, answer in enumerate(answers, start=1):
      _add_labeled_answer(body, f"h_{timestep}", answer)
    return body, answers

  @classmethod
  def _build_explanation(cls, context):
    explanation = ca.Section()
    explanation.add_element(ca.Paragraph([
      "At every timestep, the cell turns the current input into hidden-state "
      "coordinates with ",
      ca.Equation("W_{xh}x_t", inline=True),
      ", combines that with the previous hidden state through ",
      ca.Equation("W_{hh}h_{t-1}", inline=True),
      ", adds the bias, and applies tanh. The resulting ",
      ca.Equation("h_t", inline=True),
      " is the cell's new memory, which becomes the previous state for the "
      "next timestep."
    ]))
    for timestep, trace in enumerate(context.traces, start=1):
      if timestep == 1:
        reason = (
          "At the first timestep, the previous state is h_0 = "
          f"{_inline_vector([0] * context.hidden_dim)}, "
          "so the recurrent contribution begins at zero."
        )
      else:
        reason = (
          f"At timestep {timestep}, use h_{timestep - 1} from the previous "
          "step; this is how the RNN carries information forward."
        )
      explanation.add_element(ca.Paragraph([
        ca.Equation(f"t = {timestep}", inline=True),
        ": ",
        reason,
      ]))
      explanation.add_element(ca.Paragraph([
        "First map the current input into the hidden-state coordinates:",
      ]))
      explanation.add_element(ca.MathExpression([
        f"W_{{xh}}x_{timestep} = ",
        ca.Matrix(data=context.W_xh, bracket_type="b", inline=True),
        r" \cdot ",
        ca.Matrix(
          data=[[value] for value in trace["input"]],
          bracket_type="b",
          inline=True,
        ),
        " = ",
        _inline_vector(trace["input_contribution"]),
      ]))
      explanation.add_element(ca.Paragraph([
        "Then transform the previous state, which is the memory arriving from "
        "the preceding timestep:",
      ]))
      explanation.add_element(ca.MathExpression([
        f"W_{{hh}}h_{timestep - 1} = ",
        ca.Matrix(data=context.W_hh, bracket_type="b", inline=True),
        r" \cdot ",
        ca.Matrix(
          data=[[round(value, cls.ROUND_DIGITS)] for value in trace["previous"]],
          bracket_type="b",
          inline=True,
        ),
        " = ",
        _inline_vector(trace["hidden_contribution"]),
      ]))
      explanation.add_element(ca.Equation(
        f"a_{timestep} = W_{{xh}}x_{timestep} + "
        f"W_{{hh}}h_{timestep - 1} + b = "
        + _inline_vector(trace["preactivation"]),
        inline=False,
      ))
      explanation.add_element(ca.Equation(
        f"h_{timestep} = tanh(a_{timestep}) = "
        + _inline_vector(trace["state"]),
        inline=False,
      ))
    return explanation, []


@QuestionRegistry.register("cst463.sequence.embedding-lookup")
class EmbeddingLookupSequenceQuestion(SequenceForwardPassQuestion):
  """Look up a sequence of embeddings and construct a fixed-window input."""

  DEFAULT_VOCABULARY_SIZE = 5
  DEFAULT_EMBEDDING_DIM = 2
  DEFAULT_SEQUENCE_LENGTH = 3
  TOKENS = ("cat", "dog", "bird", "fish", "tree", "runs", "walks", "sleeps")

  @staticmethod
  def _generate_lookup_diagram(context) -> BytesIO:
    """Show that a token selects one row of the embedding table."""
    import matplotlib.pyplot as plt

    selected_token = context.sequence[0]
    selected_index = context.vocabulary.index(selected_token)
    rows, columns = context.embeddings.shape
    figure, axis = plt.subplots(figsize=(7.0, 0.68 * rows + 1.6))
    axis.axis("off")

    table_left = 2.0
    cell_width = 0.8
    row_height = 0.55
    top = rows * row_height
    for row, token in enumerate(context.vocabulary):
      y = top - (row + 0.5) * row_height
      highlight = "#f8d9a7" if row == selected_index else "white"
      axis.add_patch(plt.Rectangle(
        (table_left - 1.2, y - row_height / 2), 1.2, row_height,
        facecolor=highlight, edgecolor="#666666", linewidth=0.7,
      ))
      axis.text(table_left - 0.6, y, token, ha="center", va="center", fontsize=10)
      for column in range(columns):
        axis.add_patch(plt.Rectangle(
          (table_left + column * cell_width, y - row_height / 2),
          cell_width, row_height, facecolor=highlight,
          edgecolor="#666666", linewidth=0.7,
        ))
        axis.text(
          table_left + (column + 0.5) * cell_width, y,
          f"{context.embeddings[row, column]:+.1f}",
          ha="center", va="center", fontsize=10,
        )

    selected_y = top - (selected_index + 0.5) * row_height
    axis.text(0.2, selected_y, f"token: {selected_token}", ha="left",
              va="center", fontsize=11, weight="bold")
    axis.annotate(
      "", xy=(table_left - 1.22, selected_y), xytext=(1.45, selected_y),
      arrowprops={"arrowstyle": "->", "color": "#9a5b18", "lw": 1.7},
    )
    vector_x = table_left + columns * cell_width + 1.25
    axis.text(vector_x, selected_y + 0.33, "selected embedding", ha="center",
              va="bottom", fontsize=10, color="#245b80")
    axis.text(vector_x, selected_y, _inline_vector(context.embeddings[selected_index]),
              ha="center", va="center", fontsize=12, weight="bold")
    axis.annotate(
      "", xy=(vector_x - 0.55, selected_y), xytext=(table_left + columns * cell_width, selected_y),
      arrowprops={"arrowstyle": "->", "color": "#245b80", "lw": 1.7},
    )
    axis.text(table_left - 0.6, top + 0.32, "token", ha="center", va="bottom",
              fontsize=10, color="#555555")
    axis.text(table_left + columns * cell_width / 2, top + 0.32,
              "embedding table E", ha="center", va="bottom",
              fontsize=10, color="#555555")
    axis.set_xlim(0, vector_x + 1.2)
    axis.set_ylim(-0.3, top + 0.75)
    figure.tight_layout(pad=0.1)
    image = BytesIO()
    figure.savefig(image, format="png", dpi=180, bbox_inches="tight",
                   facecolor="white", edgecolor="none")
    plt.close(figure)
    image.seek(0)
    return image

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    context = super()._build_context(rng_seed=rng_seed, **kwargs)
    vocabulary_size = kwargs.get("vocabulary_size", cls.DEFAULT_VOCABULARY_SIZE)
    embedding_dim = kwargs.get("embedding_dim", cls.DEFAULT_EMBEDDING_DIM)
    sequence_length = kwargs.get("sequence_length", cls.DEFAULT_SEQUENCE_LENGTH)
    if vocabulary_size < 2 or vocabulary_size > len(cls.TOKENS):
      raise ValueError(f"vocabulary_size must be between 2 and {len(cls.TOKENS)}")
    if embedding_dim < 1 or sequence_length < 2:
      raise ValueError("embedding_dim must be positive and sequence_length at least 2")
    vocabulary = context.rng.sample(list(cls.TOKENS), vocabulary_size)
    embedding_values = tuple(value / 10 for value in range(-15, 16))
    embeddings = np.array([
      [context.rng.choice(embedding_values) for _ in range(embedding_dim)]
      for _ in range(vocabulary_size)
    ], dtype=float)
    # A repeated token makes students preserve position: the same lookup may
    # appear at more than one timestep in an embedding sequence.
    first_token, second_token = context.rng.sample(vocabulary, 2)
    sequence = [first_token]
    if sequence_length > 1:
      sequence.append(second_token)
    if sequence_length > 2:
      sequence.append(first_token)
    sequence.extend(
      context.rng.choice(vocabulary)
      for _ in range(max(0, sequence_length - len(sequence)))
    )
    token_to_index = {token: index for index, token in enumerate(vocabulary)}
    sequence_embeddings = np.array([
      embeddings[token_to_index[token]] for token in sequence
    ], dtype=float)
    context["vocabulary"] = vocabulary
    context["embeddings"] = embeddings
    context["sequence"] = sequence
    context["sequence_length"] = sequence_length
    context["sequence_embeddings"] = sequence_embeddings
    context["lookup_index"] = token_to_index[sequence[0]]
    context["input_sequence_vector"] = np.concatenate(sequence_embeddings)
    return context

  @classmethod
  def _build_body(cls, context):
    body = ca.Section()
    body.add_element(ca.Paragraph([
      "Use the embedding table to look up the vector at every position in the "
      "sequence. Then concatenate them to form the input ",
      ca.Equation("u = (x_1, \\ldots, x_T)", inline=True),
      "."
    ]))
    embedding_rows = [
      [token, *[f"{value:+.1f}" for value in embedding]]
      for token, embedding in zip(context.vocabulary, context.embeddings)
    ]
    embedding_headers = ["Token"] + [
      ca.Equation(f"e_{component + 1}", inline=True)
      for component in range(context.embeddings.shape[1])
    ]
    body.add_element(ca.Table(headers=embedding_headers, data=embedding_rows))
    body.add_element(ca.Paragraph([
      "Sequence: ",
      "[" + ", ".join(context.sequence) + "]",
    ]))

    answers = [
      ca.AnswerTypes.CompactVector(
        list(embedding),
        label="",
      )
      for timestep, embedding in enumerate(context.sequence_embeddings, start=1)
    ]
    answers.append(ca.AnswerTypes.CompactVector(
      list(context.input_sequence_vector),
      label="",
    ))
    for timestep, answer in enumerate(answers[:-1], start=1):
      _add_labeled_answer(body, f"x_{timestep}", answer)
    _add_labeled_answer(body, "u", answers[-1])
    return body, answers

  @classmethod
  def _build_explanation(cls, context):
    explanation = ca.Section()
    explanation.add_element(ca.Paragraph([
      "An embedding lookup selects the row associated with each token. The "
      "same token at two positions has the same embedding."
    ]))
    explanation.add_element(ca.Paragraph([
      "For the first position, the one-hot vector for ",
      f"“{context.sequence[0]}”",
      " has a 1 at that token's table row and 0 everywhere else. Multiplying "
      "that one-hot vector by the table keeps only the selected row.",
    ]))
    selected_embedding_rows = []
    for row, (token, embedding) in enumerate(
        zip(context.vocabulary, context.embeddings)
    ):
      cells = [token, *[f"{value:+.1f}" for value in embedding]]
      if row == context.lookup_index:
        cells = [ca.Text(cell, emphasis=True) for cell in cells]
      selected_embedding_rows.append(cells)
    embedding_headers = ["Token"] + [
      ca.Equation(f"e_{component + 1}", inline=True)
      for component in range(context.embeddings.shape[1])
    ]
    explanation.add_element(ca.Paragraph([
      "The selected row for the first token is bolded:",
    ]))
    explanation.add_element(ca.Table(
      headers=embedding_headers,
      data=selected_embedding_rows,
    ))
    for timestep, (token, embedding) in enumerate(
        zip(context.sequence, context.sequence_embeddings), start=1
    ):
      explanation.add_element(ca.Paragraph([
        f"At timestep {timestep}, token \u201c{token}\u201d selects ",
        ca.Equation(f"x_{timestep}", inline=True),
        " = ",
        _inline_vector(embedding),
        ".",
      ]))
    explanation.add_element(ca.Paragraph([
      "Concatenation places every embedding end to end in sequence order; it "
      "does not add or average them:",
    ]))
    explanation.add_element(ca.Equation(
      "u = (x_1, \\ldots, x_T) = "
      + _inline_vector(context.input_sequence_vector),
      inline=False,
    ))
    return explanation, []


@QuestionRegistry.register("cst463.cnn.output-size")
class ConvolutionOutputSizeQuestion(SequenceForwardPassQuestion):
  """Find a 2D convolution or transposed-convolution output shape."""

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    context = super()._build_context(rng_seed=rng_seed, **kwargs)
    operation = kwargs.get("operation")
    if operation is None:
      operation = context.rng.choice(("convolution", "transposed_convolution"))
    if operation not in ("convolution", "transposed_convolution"):
      raise ValueError("operation must be convolution or transposed_convolution")
    height = kwargs.get("input_height", context.rng.randint(4, 10))
    width = kwargs.get("input_width", context.rng.randint(4, 10))
    kernel = kwargs.get("kernel_size", context.rng.choice((2, 3, 4)))
    stride = kwargs.get("stride", context.rng.choice((1, 2, 3)))
    padding = kwargs.get("padding", context.rng.choice((0, 1)))
    if operation == "convolution":
      output_height = (height + 2 * padding - kernel) // stride + 1
      output_width = (width + 2 * padding - kernel) // stride + 1
      if min(output_height, output_width) < 1:
        raise ValueError("parameters must produce a positive output size")
    else:
      padding = 0
      output_height = (height - 1) * stride + kernel
      output_width = (width - 1) * stride + kernel
    context["operation"] = operation
    context["height"] = height
    context["width"] = width
    context["kernel"] = kernel
    context["stride"] = stride
    context["padding"] = padding
    context["output_height"] = output_height
    context["output_width"] = output_width
    return context

  @classmethod
  def _build_body(cls, context):
    body = ca.Section()
    operation_name = (
      "ordinary convolution"
      if context.operation == "convolution"
      else "transposed convolution"
    )
    body.add_element(ca.Paragraph([
      f"Find the output height and width of this {operation_name}.",
    ]))
    details = [
      f"Input map: {context.height}×{context.width}.",
      f"Kernel: {context.kernel}×{context.kernel}.",
      f"Stride: {context.stride}.",
    ]
    if context.operation == "convolution":
      details.append(f"Padding: {context.padding} on every side.")
    body.add_element(ca.Paragraph([" ".join(details)]))
    answer = ca.AnswerTypes.CompactVector(
      [context.output_height, context.output_width],
      label="Output shape (height, width)",
    )
    body.add_element(ca.AnswerBlock(answer))
    return body, [answer]

  @classmethod
  def _build_explanation(cls, context):
    explanation = ca.Section()
    if context.operation == "convolution":
      formula = r"\left\lfloor \frac{n + 2p - k}{s} \right\rfloor + 1"
      height_calc = (
        f"floor(({context.height} + 2({context.padding}) - {context.kernel})"
        f"/{context.stride}) + 1 = {context.output_height}"
      )
      width_calc = (
        f"floor(({context.width} + 2({context.padding}) - {context.kernel})"
        f"/{context.stride}) + 1 = {context.output_width}"
      )
      explanation.add_element(ca.Paragraph([
        "For ordinary convolution, apply the output-size formula separately "
        "to the height and width:",
      ]))
    else:
      formula = r"(n - 1) \cdot s + k"
      height_calc = (
        f"({context.height} - 1)({context.stride}) + {context.kernel}"
        f" = {context.output_height}"
      )
      width_calc = (
        f"({context.width} - 1)({context.stride}) + {context.kernel}"
        f" = {context.output_width}"
      )
      explanation.add_element(ca.Paragraph([
        "For transposed convolution, each input location stamps a kernel and "
        "the stride gives the gap between stamp origins:",
      ]))
    explanation.add_element(ca.Equation(formula, inline=False))
    explanation.add_element(ca.Equation(f"H_{{out}} = {height_calc}", inline=False))
    explanation.add_element(ca.Equation(f"W_{{out}} = {width_calc}", inline=False))
    return explanation, []


@QuestionRegistry.register("cst463.cnn.transposed-forward-pass")
class TransposedConvolutionForwardPass(SequenceForwardPassQuestion):
  """Compute a small transposed convolution by stamping a kernel."""

  @classmethod
  def _build_context(cls, *, rng_seed=None, **kwargs):
    context = super()._build_context(rng_seed=rng_seed, **kwargs)
    stride = kwargs.get("stride", 1)
    if stride < 1:
      raise ValueError("stride must be positive")
    inputs = np.array([
      [context.rng.randint(-2, 3), context.rng.randint(-2, 3)],
      [context.rng.randint(-2, 3), context.rng.randint(-2, 3)],
    ], dtype=int)
    kernel = np.array([
      [context.rng.choice((-1, 0, 1)), context.rng.choice((-1, 0, 1))],
      [context.rng.choice((-1, 0, 1)), context.rng.choice((-1, 0, 1))],
    ], dtype=int)
    output_size = (2 - 1) * stride + 2
    output = np.zeros((output_size, output_size), dtype=int)
    for row in range(2):
      for column in range(2):
        start_row, start_column = row * stride, column * stride
        output[start_row:start_row + 2, start_column:start_column + 2] += (
          inputs[row, column] * kernel
        )
    context["inputs"] = inputs
    context["kernel"] = kernel
    context["stride"] = stride
    context["output"] = output
    return context

  @classmethod
  def is_interesting_ctx(cls, context) -> bool:
    return np.any(context.output != 0) and np.unique(context.output).size >= 3

  @classmethod
  def _build_body(cls, context):
    body = ca.Section()
    body.add_element(ca.Paragraph([
      "Compute this transposed convolution. Each input value scales a copy "
      "of the kernel, then that copy is stamped into the output. Add values "
      "where stamps overlap."
    ]))
    maps = ca.TableGroup()
    maps.add_table(
      ca.Table(data=context.inputs.tolist()),
      label="Input feature map",
    )
    maps.add_table(
      ca.Table(data=context.kernel.tolist()),
      label="Kernel",
    )
    body.add_element(maps)
    body.add_element(ca.Paragraph([f"Stride: {context.stride}."]))
    placeholders = [
      [rf"y_{{{row},{column}}}" for column in range(context.output.shape[1])]
      for row in range(context.output.shape[0])
    ]
    body.add_element(ca.Paragraph(["Output map entries:"]))
    body.add_element(ca.Matrix(data=placeholders, bracket_type="b"))
    answers = []
    for row in range(context.output.shape[0]):
      for column in range(context.output.shape[1]):
        answer = ca.AnswerTypes.Int(int(context.output[row, column]), label="")
        answers.append(answer)
        _add_labeled_answer(body, f"y_{{{row},{column}}}", answer)
    return body, answers

  @classmethod
  def _build_explanation(cls, context):
    explanation = ca.Section()
    explanation.add_element(ca.Paragraph([
      "Start with an output map of zeros. For each input value, multiply the "
      "entire kernel by that value and stamp it starting at that input "
      "location times the stride. If two stamps cover a cell, their values add."
    ]))
    for row in range(2):
      for column in range(2):
        explanation.add_element(ca.Paragraph([
          f"Input value {int(context.inputs[row, column])} at ({row}, {column}) "
          f"stamps its scaled kernel beginning at output position "
          f"({row * context.stride}, {column * context.stride}):",
        ]))
        explanation.add_element(ca.Matrix(
          data=context.inputs[row, column] * context.kernel,
          bracket_type="b",
        ))
    explanation.add_element(ca.Paragraph(["After adding all stamps, the output is:"]))
    explanation.add_element(ca.Matrix(data=context.output, bracket_type="b"))
    return explanation, []
