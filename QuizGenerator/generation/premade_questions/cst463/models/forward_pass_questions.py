"""Small, hand-calculable forward-pass questions for CNNs and sequences."""

from __future__ import annotations

from io import BytesIO

import numpy as np

import QuizGenerator.generation.contentast as ca
from QuizGenerator.generation.mixins import TableQuestionMixin
from QuizGenerator.generation.question import Question, QuestionRegistry


def _number(value):
  """Format an integer or a compact decimal for an explanation."""
  if abs(value - round(value)) < 1e-10:
    return str(int(round(value)))
  return f"{value:.4f}"


def _inline_vector(values):
  """Format a vector compactly without relying on matrix rendering."""
  return "(" + ", ".join(_number(value) for value in values) + ")"


def _inline_matrix(values):
  """Format a small matrix compactly for Canvas and PDF question bodies."""
  return "[" + "; ".join(_inline_vector(row) for row in values) + "]"


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
      [context.rng.choice((0, 1)) for _ in range(input_length)]
      for _ in range(input_channels)
    ], dtype=int)
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
    ], dtype=int)

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
    return np.any(context.output != 0) and (
      context.output.size == 1 or len(set(context.output.tolist())) > 1
    )

  @classmethod
  def _build_body(cls, context):
    body = ca.Section()
    body.add_element(ca.Paragraph([
      "Compute the one-dimensional convolution output. Use cross-correlation "
      "as in the slides: do not flip the kernel."
    ]))
    body.add_element(ca.Equation(
      r"y_i = sum_c sum_a K_{c,a} x_{c, i \cdot s + a} + b",
      inline=False,
    ))

    for channel in range(context.input_channels):
      body.add_element(ca.Paragraph([
        ca.Equation(f"x_{channel + 1}", inline=True),
        " = ",
        _inline_vector(context.inputs[channel]),
        "; ",
        ca.Equation(f"K_{channel + 1}", inline=True),
        " = ",
        _inline_vector(context.kernel[channel]),
        ".",
      ]))
    body.add_element(ca.Paragraph([
      f"Stride: {context.stride}; padding: {context.padding}; "
      f"bias: {context.bias}."
    ]))

    answers = [
      ca.AnswerTypes.Int(value, label=f"y_{index}")
      for index, value in enumerate(context.output)
    ]
    output_header = ca.Equation("y_i", inline=True)
    rows = [
      {"i": str(index), output_header: answer}
      for index, answer in enumerate(answers)
    ]
    body.add_element(cls.create_answer_table(
      headers=["i", output_header],
      data_rows=rows,
      answer_columns=[output_header],
    ))
    return body, answers

  @classmethod
  def _build_explanation(cls, context):
    explanation = ca.Section()
    explanation.add_element(ca.Paragraph([
      "At each position, align every kernel slice with the matching input "
      "channel, multiply corresponding entries, then add every channel's "
      "contribution and the bias."
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
    answer = ca.AnswerTypes.Matrix(
      context.output,
      label="Pooled feature map",
    )
    body.add_element(ca.Paragraph([
      "Enter the resulting pooled feature map:",
    ]))
    body.add_element(answer)
    return body, [answer]

  @classmethod
  def _build_explanation(cls, context):
    explanation = ca.Section()
    explanation.add_element(ca.Paragraph([
      "Max pooling keeps the largest entry in each pooling window."
    ]))
    for row_index in range(context.output.shape[0]):
      for column_index in range(context.output.shape[1]):
        start_row = row_index * context.stride
        start_column = column_index * context.stride
        window = context.feature_map[
          start_row:start_row + context.pool_size,
          start_column:start_column + context.pool_size,
        ]
        values = ", ".join(str(int(value)) for value in window.flatten())
        explanation.add_element(ca.Equation(
          f"p_{{{row_index},{column_index}}} = max({values}) = "
          f"{int(context.output[row_index, column_index])}",
          inline=False,
        ))
    explanation.add_element(ca.Paragraph(["The pooled feature map is:"]))
    explanation.add_element(ca.Matrix(data=context.output.tolist(), bracket_type="b"))
    return explanation, []


@QuestionRegistry.register("cst463.rnn.state-sequence-forward-pass")
class RNNStateSequenceForwardPass(SequenceForwardPassQuestion):
  """Compute every hidden state of a tanh RNN using the course notation."""

  DEFAULT_SEQUENCE_LENGTH = 3
  DEFAULT_INPUT_DIM = 2
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
  def _generate_rnn_diagram(sequence_length: int) -> BytesIO:
    """Draw the unrolled recurrent cell shown in the course slides."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 2.25))
    axis.axis("off")
    x_positions = np.arange(sequence_length, dtype=float) * 2.25

    for index, x_position in enumerate(x_positions, start=1):
      axis.text(
        x_position, 0, rf"$x_{index}$", ha="center", va="center",
        fontsize=13, bbox={"boxstyle": "circle,pad=0.35", "fc": "#f5f5f5", "ec": "#555555"},
      )
      axis.text(
        x_position, 1.2, rf"$h_{index}$" + "\n" + "tanh",
        ha="center", va="center", fontsize=12,
        bbox={"boxstyle": "round,pad=0.45", "fc": "#d8eaf7", "ec": "#3478a6"},
      )
      axis.annotate(
        "", xy=(x_position, 0.91), xytext=(x_position, 0.28),
        arrowprops={"arrowstyle": "->", "color": "#3478a6", "lw": 1.5},
      )
      axis.text(x_position + 0.13, 0.58, r"$W_{xh}$", fontsize=9, color="#245b80")
      if index > 1:
        axis.annotate(
          "", xy=(x_position - 0.38, 1.2), xytext=(x_position - 1.87, 1.2),
          arrowprops={"arrowstyle": "->", "color": "#9a5b18", "lw": 1.7},
        )
        axis.text(
          x_position - 1.12, 1.38, r"$W_{hh}$",
          fontsize=9, color="#7c4610", ha="center",
        )

    axis.text(
      x_positions[0] - 0.8, 1.2, r"$h_0 = 0$", fontsize=11,
      ha="center", va="center",
    )
    axis.annotate(
      "", xy=(x_positions[0] - 0.38, 1.2), xytext=(x_positions[0] - 0.65, 1.2),
      arrowprops={"arrowstyle": "->", "color": "#9a5b18", "lw": 1.7},
    )
    axis.set_xlim(x_positions[0] - 1.15, x_positions[-1] + 1.1)
    axis.set_ylim(-0.55, 1.85)
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
      "Run this tanh RNN forward over the input sequence. Treat vectors as "
      "columns and use full calculator precision until rounding each submitted "
      "hidden-state component to four decimal places."
    ]))
    body.add_element(ca.Equation(
      r"h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b)",
      inline=False,
    ))
    body.add_element(ca.Picture(
      img_data=cls._generate_rnn_diagram(context.sequence_length),
      caption="Unrolled RNN: the same weights are used at every timestep.",
      width="100%",
    ))
    sequence_text = "; ".join(
      f"x_{timestep} = {_inline_vector(vector)}"
      for timestep, vector in enumerate(context.inputs, start=1)
    )
    body.add_element(ca.Paragraph(["Input sequence: ", sequence_text, "."]))
    body.add_element(ca.Paragraph([
      ca.Equation("W_{xh}", inline=True),
      " = ",
      _inline_matrix(context.W_xh),
      "; ",
      ca.Equation("W_{hh}", inline=True),
      " = ",
      _inline_matrix(context.W_hh),
      ".",
    ]))
    body.add_element(ca.Paragraph([
      ca.Equation("b", inline=True),
      " = ",
      _inline_vector(context.bias),
      ".",
    ]))
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
        label=f"h_{timestep}",
      )
      for timestep, state in enumerate(context.states, start=1)
    ]
    body.add_element(ca.AnswerBlock(answers))
    return body, answers

  @classmethod
  def _build_explanation(cls, context):
    explanation = ca.Section()
    explanation.add_element(ca.Paragraph([
      "The same weights are used at every timestep. For each state, first "
      "calculate the two linear contributions, add the bias, then apply tanh "
      "component by component."
    ]))
    for timestep, trace in enumerate(context.traces, start=1):
      explanation.add_element(ca.Paragraph([
        ca.Equation(f"t = {timestep}", inline=True), ":"
      ]))
      for component in range(context.hidden_dim):
        input_terms = " + ".join(
          f"({_number(context.W_xh[component, index])})"
          f"({_number(trace['input'][index])})"
          for index in range(context.input_dim)
        )
        hidden_terms = " + ".join(
          f"({_number(context.W_hh[component, index])})"
          f"({_number(trace['previous'][index])})"
          for index in range(context.hidden_dim)
        )
        preactivation = trace["preactivation"][component]
        state = trace["state"][component]
        explanation.add_element(ca.Equation(
          f"a_{{{timestep},{component + 1}}} = {input_terms} + "
          f"({hidden_terms}) + {_number(context.bias[component])} "
          f"= {preactivation:.4f}",
          inline=False,
        ))
        explanation.add_element(ca.Equation(
          f"h_{{{timestep},{component + 1}}} = tanh({preactivation:.4f}) "
          f"= {state:.4f}",
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
    embeddings = np.array([
      [context.rng.choice((-1, 0, 1)) for _ in range(embedding_dim)]
      for _ in range(vocabulary_size)
    ], dtype=int)
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
    ], dtype=int)
    context["vocabulary"] = vocabulary
    context["embeddings"] = embeddings
    context["sequence"] = sequence
    context["sequence_length"] = sequence_length
    context["sequence_embeddings"] = sequence_embeddings
    if sequence_length > 1:
      context["window_vector"] = np.concatenate(sequence_embeddings[:2])
    return context

  @classmethod
  def _build_body(cls, context):
    body = ca.Section()
    body.add_element(ca.Paragraph([
      "Use the embedding table to look up the vector at every position in the "
      "sequence. Then concatenate the first two embeddings as a fixed-window "
      "input. Each row of the table is that token's embedding."
    ]))
    embedding_rows = [
      [token, *[str(value) for value in embedding]]
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
      ca.AnswerTypes.Vector(
        list(embedding),
        label=f"x_{timestep}",
      )
      for timestep, embedding in enumerate(context.sequence_embeddings, start=1)
    ]
    if context.sequence_length > 1:
      answers.append(ca.AnswerTypes.Vector(
        list(context.window_vector),
        label="u = (x_1, x_2)",
      ))
    body.add_element(ca.AnswerBlock(answers))
    return body, answers

  @classmethod
  def _build_explanation(cls, context):
    explanation = ca.Section()
    explanation.add_element(ca.Paragraph([
      "An embedding lookup selects the row associated with each token. The "
      "same token at two positions has the same embedding."
    ]))
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
    if context.sequence_length > 1:
      explanation.add_element(ca.Paragraph([
        "The fixed-window representation places the first embedding followed "
        "by the second:",
      ]))
      explanation.add_element(ca.Equation(
        "u = (x_1, x_2) = "
        + _inline_vector(context.window_vector),
        inline=False,
      ))
    return explanation, []
