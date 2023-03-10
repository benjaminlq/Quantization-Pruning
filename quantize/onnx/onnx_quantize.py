import argparse
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static, quantize_dynamic, CalibrationMethod
from quantize.onnx.dataset_calibrate import CIFAR10DataReader
from utils import benchmark_inference_time
from config import LOGGER
from time import time
import os

def get_argument_parser():
    parser = argparse.ArgumentParser("ONNX Quantization")
    parser.add_argument("--onnx_ckpt", "-c", type = str, help = "Path to Float32 ONNX Model")
    parser.add_argument("--export_path", "-e", type = str, help = "Path to Quantized ONNX Model")
    parser.add_argument("--type", "-t", type = str, default = "dynamic", help = "Static | Dynamic")

    parser.add_argument("--quant_format", "-f", type = QuantFormat.from_string, default = QuantFormat.QDQ, help = "Quantization Format")
    parser.add_argument("--per_channel", action = "store_true", help = "To be added")

    # parser.add_argument("--calibrate_dataset", "-d", type = str, default = None, help = "Path to Calibration Dataset")
    parser.add_argument("--calibration_method", "-m", type = str, default = "MinMax", help = "Calibration Method a( MinMax | Entropy | Percentile )")
    
    args = parser.parse_args()
    return args

def main():
    args = get_argument_parser()
    if args.type.lower() == "static":
        LOGGER.info("Prepare Calibration dataset.")
        # assert args.calibrate_dataset is not None, "Calibration dataset needed for Static Quantization"
        datareader = CIFAR10DataReader(model_path = args.onnx_ckpt)
        LOGGER.info("Calibration Dataset loaded. Starting Static Quantization")
        start_time = time()
        quantize_static(
            args.onnx_ckpt,
            args.export_path,
            datareader,
            quant_format=args.quant_format,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=args.per_channel,
            optimize_model=False,
            calibrate_method=getattr(CalibrationMethod, args.calibration_method)
        )
        end_time = time()
        LOGGER.info(f"Static Quantization completed in {end_time - start_time} seconds.")   
        LOGGER.info(f"Benchmarking fp32 model - {benchmark_inference_time(args.onnx_ckpt)}")
        LOGGER.info(f"Benchmarking int8 model - {benchmark_inference_time(args.export_path)}")

    elif args.type.lower() == "dynamic":
        LOGGER.info("Starting Dynamic Quantization.")
        start_time = time()
        quantize_dynamic(
            args.onnx_ckpt,
            args.export_path,
            per_channel = args.per_channel,
            weight_type=QuantType.QUInt8,
            optimize_model = False,
        )
        end_time = time()
        LOGGER.info(f"Dynamic Quantization completed in {end_time - start_time} seconds.")    
        LOGGER.info(f"Benchmarking fp32 model - {benchmark_inference_time(args.onnx_ckpt)}")
        LOGGER.info(f"Benchmarking int8 model - {benchmark_inference_time(args.export_path)}")
        
    else:
        LOGGER.error("Incorrect type of quantization")
        raise TypeError("Invalid quantization config type, it must be either StaticQuantConfig or DynamicQuantConfig.")
    
    fp32_model_stats = os.stat(args.onnx_ckpt)
    int8_model_stats = os.stat(args.export_path)
    LOGGER.info(f"FP32 Model size = {fp32_model_stats.st_size}")
    LOGGER.info(f"INT8 Model size = {int8_model_stats.st_size}")
            
if __name__ =="__main__":
    main()
    
# python3 quantize/onnx/onnx_quantize.py -c models/onnx/ckpt/best_resnet50_cifar10.preproc.onnx -e models/onnx/ckpt/best_resnet50_cifar10.staticquant.onnx -t static

