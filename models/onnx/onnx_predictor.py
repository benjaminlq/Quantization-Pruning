import onnxruntime
from typing import Callable

class OnnxClassificationPredictor:
    def __init__(
        self,
        onnx_ckpt: str,
        transform: Callable,
    ):
        self.ort_session = onnxruntime(onnx_ckpt)
        
    def predict(
        self,
        imgs,
    ):
        