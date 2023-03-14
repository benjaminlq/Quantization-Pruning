import torch ## Need this to activate correct cuDNN and CUDA version

from onnxruntime.quantization.shape_inference import quant_pre_process
from onnxruntime.quantization.quantize import quantize_static, quantize_dynamic, quantize
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static, quantize_dynamic, CalibrationMethod
from quantize.onnx.dataset_calibrate import CIFAR10DataReader
from onnxruntime.quantization.qdq_loss_debug import (
    collect_activations,
    compute_activation_error,
    compute_weight_error,
    create_activation_matching,
    create_weight_matching,
    modify_model_output_intermediate_tensors
)
from utils import benchmark_inference_time, generate_aug_model_path

import argparse
import os
import os.path as osp
import onnx
from config import LOGGER
import sys
from time import time

def get_argument_parser():
    parser = argparse.ArgumentParser("ONNX Quantization")
    parser.add_argument("--source_model", "-i", type = str, help = "Path to Float32 ONNX MOdel")
    parser.add_argument("--export_path", "-o", type = str, help = "Path to Quantized ONNX Model")
    
    ### Preprocess ###
    parser.add_argument("--skip_preprocess", action = "store_true", default = False, help = "Skip all Preprocessing Steps of ONNX Model Quantization")
    parser.add_argument("--skip_optimization", action = "store_true", default = False, help = "Skip model optimization step of ONNX Model Quantization")
    parser.add_argument("--skip_onnx_shape", action = "store_true", default = False, help = "Skip ONNX Shape Inference step of ONNX Model Quantization")
    parser.add_argument("--skip_symbolic_shape", action = "store_true", default = False, help = "Skip Symbolic Shape Inference step of ONNX Model Quantization")
    parser.add_argument("--auto_merge", action="store_true", default=False, help="Automatically merge symbolic dims when confliction happens")
    parser.add_argument("--int_max", type=int, default=2**31 - 1, help="maximum value for integer to be treated as boundless for ops like slice")
    parser.add_argument("--guess_output_rank", action="store_true", default=False, help="guess output rank to be the same as input 0 for unknown ops")
    parser.add_argument("--verbose", type = int, default = 0, help="Prints detailed logs of inference, 0: turn off, 1: warnings, 3: detailed")
    parser.add_argument("--save_as_external_data", action="store_true", default=False, help="Saving an ONNX model to external data")
    parser.add_argument("--all_tensors_to_one_file", action="store_true", default=False, help="Saving all the external data to one file")
    parser.add_argument("--external_data_location", default="./", help="The file location to save the external file")
    parser.add_argument("--external_data_size_threshold", type=int, default=1024, help="The size threshold for external data")
    
    ### Quantization ###
    parser.add_argument("--type", "-t", type = str, default = "dynamic", help = "Static | Dynamic")
    parser.add_argument("--quant_format", "-f", type = QuantFormat.from_string, default = QuantFormat.QDQ, help = "Quantization Format")
    parser.add_argument("--per_channel", action = "store_true", help = "To be added")
    # parser.add_argument("--calibrate_dataset", "-d", type = str, default = None, help = "Path to Calibration Dataset")
    parser.add_argument("--calibration_method", "-m", type = str, default = "MinMax", help = "Calibration Method a( MinMax | Entropy | Percentile )")
    
    parser.add_argument("--debug", "-db", action = "store_true", default = False, help="If True, perform quantization debugging on the quantized model")
    
    args = parser.parse_args()
    return args
    
def main():
    args  = get_argument_parser()
    source_model = args.source_model
    if args.skip_preprocess or (args.skip_optimization and args.skip_onnx_shape and args.skip_symbolic_shape):
        LOGGER.info("Skip ONNX Model preprocessing stage")
        pre_process_model = source_model
    else:
        if (not args.skip_optimization) and args.save_as_external_data:
            LOGGER.error("ORT model optimization does not support external data yet!")
            sys.exit()
        pre_process_model = osp.splitext(source_model)[0] + ".tmppreproc.onnx"
        LOGGER.info("Start Model Preprocessing Stage")
        LOGGER.info(f"Source Model Path: {source_model}")
        LOGGER.info(f"Preprocessed Model Path: {pre_process_model}")
        quant_pre_process(
            source_model,
            pre_process_model,
            args.skip_optimization,
            args.skip_onnx_shape,
            args.skip_symbolic_shape,
            args.auto_merge,
            args.int_max,
            args.guess_output_rank,
            args.verbose,
            args.save_as_external_data,
            args.all_tensors_to_one_file,
            args.external_data_location,
            args.external_data_size_threshold,   
        )
    
    calibration_datareader = CIFAR10DataReader(model_path = pre_process_model)
    
    if args.type.lower() == "static":
        LOGGER.info("Prepare Calibration dataset.")
        # assert args.calibrate_dataset is not None, "Calibration dataset needed for Static Quantization"
        LOGGER.info("Calibration Dataset loaded. Starting Static Quantization")
        start_time = time()
        quantize_static(
            pre_process_model,
            args.export_path,
            calibration_datareader,
            quant_format=args.quant_format,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=args.per_channel,
            optimize_model=False,
            calibrate_method=getattr(CalibrationMethod, args.calibration_method)
        )
        end_time = time()
        LOGGER.info(f"Static Quantization completed in {end_time - start_time} seconds.")   
        LOGGER.info(f"Benchmarking fp32 model - {benchmark_inference_time(pre_process_model)}")
        LOGGER.info(f"Benchmarking int8 model - {benchmark_inference_time(args.export_path)}")

    elif args.type.lower() == "dynamic":
        LOGGER.info("Starting Dynamic Quantization.")
        start_time = time()
        quantize_dynamic(
            pre_process_model,
            args.export_path,
            per_channel = args.per_channel,
            weight_type=QuantType.QUInt8,
            optimize_model = False,
        )
        end_time = time()
        LOGGER.info(f"Dynamic Quantization completed in {end_time - start_time} seconds.")    
        LOGGER.info(f"Benchmarking fp32 model - {benchmark_inference_time(pre_process_model)}")
        LOGGER.info(f"Benchmarking int8 model - {benchmark_inference_time(args.export_path)}")
        
    else:
        LOGGER.error("Incorrect type of quantization")
        raise TypeError("Invalid quantization config type, it must be either StaticQuantConfig or DynamicQuantConfig.")
    
    fp32_model_stats = os.stat(args.source_model)
    int8_model_stats = os.stat(args.export_path)
    LOGGER.info(f"FP32 Model size = {fp32_model_stats.st_size}")
    LOGGER.info(f"INT8 Model size = {int8_model_stats.st_size}")
    
    if args.debug:
        LOGGER.info("Start Quantization Debugging Process")    
        LOGGER.info("Comparing weights of float model vs qdq model.....")
        matched_weights = create_weight_matching(float_model_path = pre_process_model, qdq_model_path=args.export_path)
        weight_errors = compute_weight_error(matched_weights)
        for weight_name, err in weight_errors.items():
            LOGGER.info(f"Cross model error of '{weight_name}': {err}\n")
        LOGGER.info("Augmenting models to save intermediate activations......")
        
        aug_float_model = modify_model_output_intermediate_tensors(pre_process_model)
        aug_float_model_path = generate_aug_model_path(pre_process_model)
        onnx.save(
            aug_float_model, aug_float_model_path, save_as_external_data=False
            )
        del aug_float_model
        
        aug_qdq_model = modify_model_output_intermediate_tensors(args.export_path)
        aug_qdq_model_path = generate_aug_model_path(args.export_path)
        onnx.save(
            aug_qdq_model, aug_qdq_model_path, save_as_external_data=False
        )
        del aug_qdq_model
        
        LOGGER.info("Running augmented float model and qdq model to collect activations")
        calibration_datareader.rewind()
        float_activations = collect_activations(augmented_model=aug_float_model_path, input_reader=calibration_datareader)
        calibration_datareader.rewind()
        qdq_activations = collect_activations(
            augmented_model=aug_qdq_model_path,
            input_reader=calibration_datareader
        )
        
        LOGGER.info("Comparing activations of augmented float and qdq models to collect activation matching")
        act_matching = create_activation_matching(qdq_activations=qdq_activations, float_activations=float_activations)
        act_errors = compute_activation_error(act_matching)
        for act_name, err in act_errors.items():
            LOGGER.info(f"Cross model error of '{act_name}': {err['xmodel_err']} \n")
            LOGGER.info(f"QDQ error of '{act_name}': {err['qdq_err']} \n")
        
if __name__ == "__main__":
    main()    

# python3 quantize/onnx/onnx_pipeline.py -i models/onnx/ckpt/best_resnet50_cifar10.onnx -o models/onnx/ckpt/best_resnet50_cifar10.staticquantfullpipe.onnx -t static
# python3 quantize/onnx/onnx_pipeline.py -i models/onnx/ckpt/best_resnet50_cifar10.onnx -o models/onnx/ckpt/best_resnet50_cifar10.dynamicquantfullpipe.onnx -t dynamic