# Model Optimization and Quantization with ONNX models

## Quantization Steps
1. Pre-Processing
Pre-processing is to transform a float32 model to prepare it for quantization. It consists of the following three optional steps:
- Symbolic shape inference: Figure out Tensor shapes. Best for Transformer Models.
- Model Optimization: Rewrite Computational Graph (merge computation nodes, eliminate redundancies to improve runtime efficiency)
- ONNX shape inference: Figure out Tensor shapes. Work best for other models.
```
python -m onnxruntime.quantization.preprocess --input <Input Float32 ONNX model> --output <Output ONNX Model Dỉrectory>
```

2. Quantization

3. Debugging