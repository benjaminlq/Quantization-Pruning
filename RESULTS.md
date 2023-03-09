# Results Summary

# CIFAR10 Classification

## Performance (Accuracy %)
|   **Quantization**   |     **Pytorch**    |     **ONNX**     |   **Tensorflow**  |
|    :-------------:   |  :--------------:  | :--------------: | :---------------: |
|  Original (Float32)  |      **96.07 %**    |    **96.07 %**    |      **xx.xx%**      |
|       Float16        |     **xx.xx%**     |    **xx.xx%**    |      **xx.xx%**      |
|         Int8         |     **xx.xx%**     |    **xx.xx%**    |      **xx.xx%**      |
|     Dynamic Range    |     **xx.xx%**     |    **xx.xx%**    |      **xx.xx%**      |

<br>

## Size (MB)
|   **Quantization**   |     **Pytorch**    |     **ONNX**     |   **Tensorflow**  |
|    :-------------:   |  :--------------:  | :--------------: | :---------------: |
|  Original (Float32)  |      **94.4 MB**      |      **94.1 MB**     |      **xx.xx MB**      |
|       Float16        |     **xx.xx**     |    **xx.xx**    |      **xx.xx**      |
|         Int8         |     **xx.xx**     |    **xx.xx**    |      **xx.xx**      |
|     Dynamic Range    |     **xx.xx**     |    **xx.xx**    |      **xx.xx**      |

<br>

## Speed (seconds)
- 1 epoch on CIFAR10 validation dataset

### CPU
|   **Quantization**   |     **Pytorch**    |     **ONNX**     |   **Tensorflow**  |
|    :-------------:   |  :--------------:  | :--------------: | :---------------: |
|  Original (Float32)  |      **702.12 s**    |      **404.67 s**     |      **xx.xx s**      |
|       Float16        |     **xx.xx**     |    **xx.xx**    |      **xx.xx**      |
|         Int8         |     **xx.xx**     |    **xx.xx**    |      **xx.xx**      |
|     Dynamic Range    |     **xx.xx**     |    **xx.xx**    |      **xx.xx**      |

### GPU
|   **Quantization**   |     **Pytorch**    |     **ONNX**     |   **Tensorflow**  |
|    :-------------:   |  :--------------:  | :--------------: | :---------------: |
|  Original (Float32)  |      **55.13 s**    |      **37.05 s**     |      **xx.xx s**      |
|       Float16        |     **xx.xx**     |    **xx.xx**    |      **xx.xx**      |
|         Int8         |     **xx.xx**     |    **xx.xx**    |      **xx.xx**      |
|     Dynamic Range    |     **xx.xx**     |    **xx.xx**    |      **xx.xx**      |