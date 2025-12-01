import abc
import logging
import math
import keras
import numpy as np

from QuizGenerator.question import Question, QuestionRegistry, Answer
from QuizGenerator.contentast import ContentAST
from QuizGenerator.constants import MathRanges

log = logging.getLogger(__name__)


@QuestionRegistry.register("cst463.cnn.convolution")
class ConvolutionCalcuation(Question):
  
  @staticmethod
  def conv2d_multi_channel(image, kernel, stride=1, padding=0):
    """
    image: (H, W, C_in) - height, width, input channels
    kernel: (K_h, K_w, C_in, C_out) - kernel height, width, input channels, output filters
    Returns: (H_out, W_out, C_out)
    """
    H, W, C_in = image.shape
    K_h, K_w, _, C_out = kernel.shape
    
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
          receptive_field = image[h_start:h_start + K_h, w_start:w_start + K_w, :]
          output[i, j, f] = np.sum(receptive_field * kernel[:, :, :, f])
    
    return output
  
  def refresh(self, *args, **kwargs):
    super().refresh(*args, **kwargs)
    self.rng = np.random.RandomState(kwargs.get("rng_seed", None))
    
    num_input_channels = 1
    
    # Small sizes for hand calculation
    self.image = self.rng.randint(0, 3, size=(4, 4, num_input_channels))  # 4x4 image, 2 channels
    self.kernel = self.rng.randint(-1, 2, size=(3, 3, num_input_channels, 3))  # 3x3 kernel, 2 in channels, 3 filters
    
    self.result = self.conv2d_multi_channel(self.image, self.kernel, stride=1, padding=0)
    
    
    return True
    
  def get_body(self, **kwargs) -> ContentAST.Section:
    body = ContentAST.Section()
    
    body.add_elements([
      ContentAST.Text("Given image represented as matrix: "),
      ContentAST.Matrix(self.image)
    ])

    body.add_elements([
      ContentAST.Text("And convolution filters: "),
      ] + [
        ContentAST.Matrix(self.kernel[:,:,:,i])
        for i in range(self.kernel.shape[-1])
      ]
    )
    
    body.add_element(
      ContentAST.Paragraph([
        "Calculate the output of the convolution operation.  Assume stride = 1 and padding = 0."
      ])
    )
    
    return body
  
  def get_explanation(self, **kwargs) -> ContentAST.Section:
    explanation = ContentAST.Section()
    
    return explanation

