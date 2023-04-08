import argparse
import onnx
from onnxruntime.quantization.qdq_loss_debug import (
    collect_activations,
    compute_activation_error,
    compute_weight_error,
    create_activation_matching,
    create_weight_matching,
    modify_model_output_intermediate_tensors
)
from config import LOGGER

from quantize.onnx.dataset_calibrate import CIFAR10DataReader

def get_argument_parser():
    parser = argparse.ArgumentParser("QUantization Debugging")
    parser.add_argument("--float_model", "-f", required=True, type=str, help="Path to original FP model")
    parser.add_argument("--qdq_model", "-q", required=True, type=str, help="Path to quantized QDQ INT8 model")
    
    args = parser.parse_args()
    return args
    
def generate_aug_model_path(model_path: str) -> str:
    aug_model_path = (
        model_path[: -len(".onnx")] if model_path.endswith(".onnx") else model_path
    )
    return aug_model_path + ".save_tensors.onnx"

def main():
    args = get_argument_parser()
    LOGGER.info("Start Quantization Debugging Process")
    
    LOGGER.info("Comparing weights of float model vs qdq model.....")
    matched_weights = create_weight_matching(float_model_path = args.float_model, qdq_model_path=args.qdq_model)
    weight_errors = compute_weight_error(matched_weights)
    for weight_name, err in weight_errors.items():
        LOGGER.info(f"Cross model error of '{weight_name}': {err}\n")
        
    LOGGER.info("Augmenting models to save intermediate activations......")
    aug_float_model = modify_model_output_intermediate_tensors(args.float_model)
    aug_float_model_path = generate_aug_model_path(args.float_model)
    onnx.save(
        aug_float_model, aug_float_model_path, save_as_external_data=False
    )
    del aug_float_model
    
    aug_qdq_model = modify_model_output_intermediate_tensors(args.qdq_model)
    aug_qdq_model_path = generate_aug_model_path(args.qdq_model)
    onnx.save(
        aug_qdq_model, aug_qdq_model_path, save_as_external_data=False
    )
    del aug_qdq_model
    
    LOGGER.info("Running augmented float model and qdq model to collect activations")
    calibration_data_reader = CIFAR10DataReader(model_path = args.float_model, batch_size = 16, batch_no=96)
    float_activations = collect_activations(augmented_model=aug_float_model_path, input_reader=calibration_data_reader)
    calibration_data_reader.rewind()
    qdq_activations = collect_activations(
        augmented_model=aug_qdq_model_path,
        input_reader=calibration_data_reader
    )
    
    LOGGER.info("Comparing activations of augmented float and qdq models to collect activation matching")
    act_matching = create_activation_matching(qdq_activations=qdq_activations, float_activations=float_activations)
    act_errors = compute_activation_error(act_matching)
    for act_name, err in act_errors.items():
        LOGGER.info(f"Cross model error of '{act_name}': {err['xmodel_err']} \n")
        LOGGER.info(f"QDQ error of '{act_name}': {err['qdq_err']} \n")
    
    
if __name__ == "__main__":
    main()

# python3 quantize/onnx/onnx_debug.py --float_model models/onnx/ckpt/best_resnet50_cifar10.preproc.onnx --qdq_model models/onnx/ckpt/best_resnet50_cifar10.staticquant.onnx