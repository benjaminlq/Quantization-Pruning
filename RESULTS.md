# Results Summary

# CIFAR10 Classification

## Performance (Accuracy %)
|   **Quantization**   |     **Pytorch**    |     **ONNX**     |   **Tensorflow**  |
|    :-------------:   |  :--------------:  | :--------------: | :---------------: |
|  Original (Float32)  |      **96.07 %**    |    **96.07 %**    |      **xx.xx%**      |
|       Float16        |     **xx.xx%**     |    **Not Supported**    |      **xx.xx%**      |
|         Int8         |     **xx.xx%**     |    **94.35%***    |      **xx.xx%**      |
|     Dynamic Range    |     **xx.xx%**     |    **95.56%**    |      **xx.xx%**      |

**\* :** Accuracy of **Static Quantization** may be different for teach quantization run depending on the calibration data sample provided. For small sample, accuracy difference may be up to **2-3%**.

<br> 644.98

## Size (MB)
|   **Quantization**   |     **Pytorch**    |     **ONNX**     |   **Tensorflow**  |
|    :-------------:   |  :--------------:  | :--------------: | :---------------: |
|  Original (Float32)  |      **94.4 MB**      |      **94.1 MB**     |      **xx.xx MB**      |
|       Float16        |     **xx.xx**     |    **Not Supported**    |      **xx.xx**      |
|         Int8         |     **xx.xx**     |    **23.70**    |      **xx.xx**      |
|     Dynamic Range    |     **xx.xx**     |    **23.70**    |      **xx.xx**      |

<br>

## Speed (seconds)
- 1 epoch on CIFAR10 validation dataset

### CPU
|   **Quantization**   |     **Pytorch**    |     **ONNX**     |   **Tensorflow**  |
|    :-------------:   |  :--------------:  | :--------------: | :---------------: |
|  Original (Float32)  |      **702.12 s**    |      **404.67 s**     |      **xx.xx s**      |
|       Float16        |     **xx.xx**     |    **Not Supported**    |      **xx.xx**      |
|         Int8         |     **xx.xx**     |    **272.04 s**    |      **xx.xx**      |
|     Dynamic Range    |     **xx.xx**     |    **754.32 s**    |      **xx.xx**      |

### GPU
|   **Quantization**   |     **Pytorch**    |     **ONNX**     |   **Tensorflow**  |
|    :-------------:   |  :--------------:  | :--------------: | :---------------: |
|  Original (Float32)  |      **55.13 s**    |      **37.05 s**     |      **xx.xx s**      |
|       Float16        |     **xx.xx**     |    **Not Supported**    |      **xx.xx**      |
|         Int8         |     **xx.xx**     |    **63.02 s**    |      **xx.xx**      |
|     Dynamic Range    |     **xx.xx**     |    **644.98 s**    |      **xx.xx**      |

* **ONNX**: Quantization does not improvement performance (Inference Time) as GPU model does not support Tensor Core INT8 computation (T4 or A100). To test preformance benchmark on Tensor Core INT8 supported GPU (Turing-T, Ampere-A, Hopper-H series supported INT8 & INT4 Tensor Core computation)
