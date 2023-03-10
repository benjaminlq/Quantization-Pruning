from onnxruntime.quantization.shape_inference import quant_pre_process
from onnxruntime.quantization.quantize import quantize_static, quantize_dynamic, quantize
 
import argparse
import os.path as osp
from config import LOGGER
import sys

def get_argument_parser():
    parser = argparse.ArgumentParser("ONNX Quantization")
    parser.add_argument("--source_model", "-i", type = str, help = "Path to Float32 ONNX MOdel")
    parser.add_argument("--type", "-t", type = str, help = "dynamic | static")
    
    parser.add_argument("--skip_preprocess", action = "store_true", default = False, help = "Skip all Preprocessing Steps of ONNX Model Quantization")
    parser.add_argument("--skip_optimization", action = "store_true", default = False, help = "Skip model optimization step of ONNX Model Quantization")
    parser.add_argument("--skip_onnx_shape", action = "store_true", default = False, help = "Skip ONNX Shape Inference step of ONNX Model Quantization")
    parser.add_argument("--skip_symbolic_shape", action = "store_true", default = False, help = "Skip Symbolic Shape Inference step of ONNX Model Quantization")
    parser.add_argument("--auto_merge", action="store_true", default=False, help="Automatically merge symbolic dims when confliction happens")
    parser.add_argument("--int_max", type=int, default=2**31 - 1, help="maximum value for integer to be treated as boundless for ops like slice")
    parser.add_argument("--guess_output_rank", action="store_true", default=False, help="guess output rank to be the same as input 0 for unknown ops")
    parser.add_argument("--verbose", type = int, default = 0, help="Prints detailed logs of inference, 0: turn off, 1: warnings, 3: detailed")
    parser.add_argument(
        "--save_as_external_data",
        help="Saving an ONNX model to external data",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--all_tensors_to_one_file",
        help="Saving all the external data to one file",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--external_data_location",
        help="The file location to save the external file",
        default="./",
    )
    parser.add_argument(
        "--external_data_size_threshold",
        help="The size threshold for external data",
        type=int,
        default=1024,
    )
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
        pre_process_model = osp.splitext(source_model)[0] + ".preproc.onnx"
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