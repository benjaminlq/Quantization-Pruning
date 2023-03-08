# Results Summary

# CIFAR10 Classification

## Performance (Accuracy - %)
|   **Quantization**   |     **Pytorch**    |     **ONNX**     |   **Tensorflow**  |
|    :-------------:   |  :--------------:  | :--------------: | :---------------: |
|  Original (Float32)  |      **92.01%**    |    **91.87%**    |      **xx.xx%**      |
|       Float16        |     **xx.xx%**     |    **xx.xx%**    |      **xx.xx%**      |
|         Int8         |     **xx.xx%**     |    **xx.xx%**    |      **xx.xx%**      |
|     Dynamic Range    |     **xx.xx%**     |    **xx.xx%**    |      **xx.xx%**      |

<br>

## Size (MB)
|   **Quantization**   |     **Pytorch**    |     **ONNX**     |   **Tensorflow**  |
|    :-------------:   |  :--------------:  | :--------------: | :---------------: |
|  Original (Float32)  |      **99.4**      |      **94.1**     |      **xx.xx**      |
|       Float16        |     **xx.xx**     |    **xx.xx**    |      **xx.xx**      |
|         Int8         |     **xx.xx**     |    **xx.xx**    |      **xx.xx**      |
|     Dynamic Range    |     **xx.xx**     |    **xx.xx**    |      **xx.xx**      |

<br>

## Speed (seconds - 1 validation epoch CIFAR10)

### CPU
|   **Quantization**   |     **Pytorch**    |     **ONNX**     |   **Tensorflow**  |
|    :-------------:   |  :--------------:  | :--------------: | :---------------: |
|  Original (Float32)  |      **665.91**    |      **320.29%**     |      **xx.xx**      |
|       Float16        |     **xx.xx**     |    **xx.xx**    |      **xx.xx**      |
|         Int8         |     **xx.xx**     |    **xx.xx**    |      **xx.xx**      |
|     Dynamic Range    |     **xx.xx**     |    **xx.xx**    |      **xx.xx**      |

### GPU
|   **Quantization**   |     **Pytorch**    |     **ONNX**     |   **Tensorflow**  |
|    :-------------:   |  :--------------:  | :--------------: | :---------------: |
|  Original (Float32)  |      **665.91**    |      **320.29%**     |      **xx.xx**      |
|       Float16        |     **xx.xx**     |    **xx.xx**    |      **xx.xx**      |
|         Int8         |     **xx.xx**     |    **xx.xx**    |      **xx.xx**      |
|     Dynamic Range    |     **xx.xx**     |    **xx.xx**    |      **xx.xx**      |