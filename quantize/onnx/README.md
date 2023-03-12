# Model Optimization and Quantization with ONNX models

## Quantization Steps
### 1. Pre-Processing
Pre-processing is to transform a float32 model to prepare it for quantization. It consists of the following three optional steps:
- Symbolic shape inference: Figure out Tensor shapes. Best for Transformer Models.
- Model Optimization: Rewrite Computational Graph (merge computation nodes, eliminate redundancies to improve runtime efficiency)
- ONNX shape inference: Figure out Tensor shapes. Work best for other models.
```
python -m onnxruntime.quantization.preprocess --input <Input Float32 ONNX model> --output <Output ONNX Model Dỉrectory>
```

Useful Arguments:
- `--skip_optimization`: Skip model optimization step if **True**. It's a known issue that ORT optimization has difficulty with model size greater than 2GB, rerun with this option to get around this issue.
- `--skip_onnx_shape`: Skip ONNX shape inference. 
- `--skip_symbolic_shape`: Skip symbolic shape inference.
- `--auto_merge`: Automatically merge symbolic dims when confliction happens.
- `--save_as_external_data`: Saving an ONNX model to external data.
- `--all_tensors_to_one_file`: Saving all the external data to one file.
- `--external_data_location`: File Location to save the external file.
- `--external_data_size_threshold`: The size threshold for external data.

### 2. Quantization


### 3. Debugging