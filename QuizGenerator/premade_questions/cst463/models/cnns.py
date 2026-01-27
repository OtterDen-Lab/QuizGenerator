import abc
import logging
import math
import keras
import numpy as np
from typing import List, Tuple

from QuizGenerator.question import Question, QuestionRegistry
import QuizGenerator.contentast as ca
from QuizGenerator.constants import MathRanges
from .matrices import MatrixQuestion

log = logging.getLogger(__name__)


@QuestionRegistry.register("cst463.cnn.convolution")
class ConvolutionCalculation(MatrixQuestion):
  
  @staticmethod
  def conv2d_multi_channel(image, kernel, stride=1, padding=0):
    """
    image: (H, W, C_in) - height, width, input channels
    kernel: (K_h, K_w, C_in, C_out) - kernel height, width, input channels, output filters
    Returns: (H_out, W_out, C_out)
    """
    H, W = image.shape
    K_h, K_w, C_out = kernel.shape
    
    # Add padding
    if padding > 0:
      image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
      H, W = H + 2 * padding, W + 2 * padding
    
    # Output dimensions
    H_out = (H - K_h) // stride + 1
    W_out = (W - K_w) // stride + 1
    
    output = np.zeros((H_out, W_out, C_out))
    
    # Convolve each filter
    for f in range(C_out):
      for i in range(H_out):
        for j in range(W_out):
          h_start = i * stride
          w_start = j * stride
          # Extract receptive field and sum over all input channels
          receptive_field = image[h_start:h_start + K_h, w_start:w_start + K_w]
          output[i, j, f] = np.sum(receptive_field * kernel[:, :, f])
    
    return output
  
  def refresh(self, *args, **kwargs):
    super().refresh(*args, **kwargs)

    # num_input_channels = 1
    input_size = kwargs.get("input_size", 4)
    num_filters = kwargs.get("num_filters", 1)
    self.stride = kwargs.get("stride", 1)
    self.padding = kwargs.get("padding", 0)

    # Small sizes for hand calculation
    self.image = self.get_rounded_matrix((input_size, input_size))
    self.kernel = self.get_rounded_matrix((3, 3, num_filters), -1, 1)

    self.result = self.conv2d_multi_channel(self.image, self.kernel, stride=self.stride, padding=self.padding)
    
    self.answers = {
      f"result_{i}" : ca.AnswerTypes.Matrix(self.result[:,:,i], label=f"Result of filter {i}")
      for i in range(self.result.shape[-1])
    }
    
    return True
  
  def _get_body(self, **kwargs) -> Tuple[ca.Section, List[ca.Answer]]:
    """Build question body and collect answers."""
    body = ca.Section()
    answers = []

    body.add_elements(
      [
        ca.Text("Given image represented as matrix: "),
        ca.Matrix(self.image, name="image")
      ]
    )

    body.add_elements(
      [
        ca.Text("And convolution filters: "),
      ] + [
        ca.Matrix(self.kernel[:, :, i], name=f"Filter {i}")
        for i in range(self.kernel.shape[-1])
      ]
    )

    body.add_element(
      ca.Paragraph(
        [
          f"Calculate the output of the convolution operation.  Assume stride = {self.stride} and padding = {self.padding}."
        ]
      )
    )

    body.add_element(ca.LineBreak())

    for i in range(self.result.shape[-1]):
      answers.append(self.answers[f"result_{i}"])
      body.add_elements([
        ca.Container([
          self.answers[f"result_{i}"],
          ca.LineBreak()
        ])
      ])

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
        "To compute a 2D convolution, we slide the filter across the input image and compute the element-wise product at each position, then sum the results."
      ])
    )

    explanation.add_element(
      ca.Paragraph([
        f"With stride={self.stride} and padding={self.padding}: ",
        f"stride controls how many pixels the filter moves each step, ",
        f"and padding adds zeros around the border {'(no border in this case)' if self.padding == 0 else f'({self.padding} pixels)'}."
      ])
    )

    # For each filter, show one detailed example computation
    for f_idx in range(self.kernel.shape[-1]):
      explanation.add_element(
        ca.Paragraph([
          ca.Text(f"Filter {f_idx}:", emphasis=True)
        ])
      )

      # Show the filter (rounded)
      explanation.add_element(
        ca.Matrix(np.round(self.kernel[:, :, f_idx], digits), name=f"Filter {f_idx}")
      )

      # Show ONE example computation (position 0,0)
      explanation.add_element(
        ca.Paragraph([
          "Example computation at position (0, 0):"
        ])
      )

      # Account for padding when extracting receptive field
      if self.padding > 0:
        padded_image = np.pad(self.image, ((self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        receptive_field = padded_image[0:3, 0:3]
      else:
        receptive_field = self.image[0:3, 0:3]

      computation_steps = []
      for r in range(3):
        row_terms = []
        for c in range(3):
          img_val = receptive_field[r, c]
          kernel_val = self.kernel[r, c, f_idx]
          row_terms.append(f"({img_val:.2f} \\times {kernel_val:.2f})")
        computation_steps.append(" + ".join(row_terms))

      equation_str = " + ".join(computation_steps)
      result_val = self.result[0, 0, f_idx]

      explanation.add_element(
        ca.Equation(f"{equation_str} = {result_val:.2f}")
      )

      # Show the complete output matrix (rounded)
      explanation.add_element(
        ca.Paragraph([
          "Complete output:"
        ])
      )
      explanation.add_element(
        ca.Matrix(np.round(self.result[:, :, f_idx], digits))
      )

    return explanation, []

  def get_explanation(self, **kwargs) -> ca.Section:
    """Build question explanation (backward compatible interface)."""
    explanation, _ = self._get_explanation(**kwargs)
    return explanation
