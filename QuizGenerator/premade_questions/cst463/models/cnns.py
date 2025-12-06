import abc
import logging
import math
import keras
import numpy as np

from QuizGenerator.question import Question, QuestionRegistry
from QuizGenerator.misc import Answer, MatrixAnswer
from QuizGenerator.contentast import ContentAST
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
      image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
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
    
    # Small sizes for hand calculation
    self.image = self.get_rounded_matrix((input_size, input_size))
    self.kernel = self.get_rounded_matrix((3, 3, num_filters), -1, 1)
    
    self.result = self.conv2d_multi_channel(self.image, self.kernel, stride=1, padding=0)
    
    self.answers = {
      f"result_{i}" : MatrixAnswer(f"result_{i}", self.result[:,:,i])
      for i in range(self.result.shape[-1])
    }
    
    return True
  
  def get_body(self, **kwargs) -> ContentAST.Section:
    body = ContentAST.Section()
    
    body.add_elements(
      [
        ContentAST.Text("Given image represented as matrix: "),
        ContentAST.Matrix(self.image, name="image")
      ]
    )
    
    body.add_elements(
      [
        ContentAST.Text("And convolution filters: "),
      ] + [
        ContentAST.Matrix(self.kernel[:, :, i], name=f"Filter {i}")
        for i in range(self.kernel.shape[-1])
      ]
    )
    
    body.add_element(
      ContentAST.Paragraph(
        [
          "Calculate the output of the convolution operation.  Assume stride = 1 and padding = 0."
        ]
      )
    )
    
    body.add_elements([
      self.answers[f"result_{i}"].get_ast_element(label=f"Result of filter {i}")
      for i in range(self.result.shape[-1])
    ]
    )
    
    
    
    return body
  
  def get_explanation(self, **kwargs) -> ContentAST.Section:
    explanation = ContentAST.Section()
    
    return explanation
