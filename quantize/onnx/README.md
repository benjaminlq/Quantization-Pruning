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
#### **Dynamic Quantization**

Only weight quantization parameters (scale and zero points) during post training quantization. Activation quantization parameters are calculated during inference dynamically. 

- PRO: Parameters are estimated more accurately on inference data, thus inference accuracy is usually higher
- CON: Slow inference speed due to additional calculation required on quantization parameters.
- CON: Some operators (Conv) only support QUInt8 (unsigned INT8) which results in significant slower inference speed compared to QInt8 (signed INT8).

#### **Static Quantization**

Both weight and activation quantization parameters are pre-calculated during post-training quantization. The activation parameters are calculated using a calibration dataset and are kept constant during inference. Two type of calibration methods: MinMax & Entropy.

- PRO: Faster Inference Speed compared to Dynamic Quantization as activation parameters are pre-calculated.
- PRO: Weight and Activation types can be set to QINT8.
- CON: Accuracy is generally lower than dynamic as pre-calculated activation parameters may not be best for inference data.
- CON: Calibration data required and need to write more codes. 

### 3. Debugging

Due to loss of precision, it can be expected that model accuracy may be reduced after quantization. Debugging identify the operations which have the largest difference between pre and post-quantized model. Based on this information, the nodes with significant difference may be skipped during quantization to preserve model accuracy.

#### **Note: For debugging to work, the graph of floating point model and qdq model must match.**

<br>

#### **Weights Matching**
- `create_weight_matching` takes a float32 model and its quantized model, and output a dictionary that matches the corresponding weights between these two models.
- `compute_weight_error` takes a set of weight matching and calculate the error score (SQNR-see below) indicating the difference between float model weights and qdq model weights.
<br>

#### **Activations Matching**
- `modify_model_output_intermediate_tensors` takes a float32 or quantized model, and augment it to save all its activations.
- `collect_activations` takes a model augmented by modify_model_output_intermediate_tensors(), and an input data reader, runs the augmented model to collect all the activations.
- `create_activation_matching` takes these two set of activations, and matches up corresponding ones, so that they can be easily compared by the user
- `compute_activation_error` akes a set of activation matching and calculate the error score (SQNR-see below) indicating the difference between augmented float model activations and augmented qdq model activations.

<br>

#### **Matching Metric**
Signal to quantization noise ratio (SQNR) is a measure of the quality of the quantization, or digital conversion of an analog signal.

**NOTE: A higher SQNR score indicates a lower quantization error.**


- General Formula
$$ SQNR = \frac{Normalized Signal Power}{Normalized Quantization Noise Power} $$

<br>

- ONNX Implementation

$$ SQNR = 20 * log(\frac{||float\_params||_2}{||float\_params-qdq\_params||_2}) $$


