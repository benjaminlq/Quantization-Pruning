import torch
import torchvision
from typing import Tuple, List, Optional
import os
import argparse
from config import LOGGER
from time import time
import onnxruntime as ort
from utils import to_numpy
import numpy as np
from models.pytorch.model_resnet import PretrainedResNet50

def get_argument_parser():
    parser = argparse.ArgumentParser("Torch-to-ONNX conversion")
    parser.add_argument("--torch_ckpt", "-t", type = str, help = "Path to Torch Model")
    parser.add_argument("--export_path", "-x", type = str, help = "Export Path to store Onnx Model")
    parser.add_argument("--no_dynamic_batch", action = "store_false", help = "Whether Onnx Model takes dynamic batch size")
    parser.add_argument("--no_const_folding", action = "store_false", help = "Whether Apply Constant Folding")
    parser.add_argument("--op_version", type = int, default = 11, help = "Version of ONNX opset")
    args = parser.parse_args()
    return args
    
def pytorch_to_onnx(
    torch_model,
    onnx_ckpt: str,
    torch_ckpt: Optional[str] = None,
    input_size: Tuple = (3, 224, 224),
    input_names: List = ["inputs"],
    output_names: List = ["outputs"],
    constant_folding: bool = True,
    dynamic_batch: bool = True,
    op_version: int = 11,
):
    
    if torch_ckpt:
        if not os.path.exists(torch_ckpt):
            raise Exception("Model Path does not exist")
        # torch_model.load_state_dict(torch.load(torch_ckpt))
        torch_model.load_state_dict(torch.load(torch_ckpt))
        LOGGER.info(f"Finished loading model {str(torch_model)}")
    dummy_input = torch.randn(1, *input_size, requires_grad=True)
    torch_model.eval()
    sample_output = torch_model(dummy_input)
    dynamic_batch = {"inputs": {0: "batch_size"},
                     "outputs": {0: "batch_size"}} if dynamic_batch else None
    LOGGER.info("Start Converting Pytorch Model to Onnx")
    torch.onnx.export(
        torch_model,
        dummy_input,
        onnx_ckpt,
        export_params=True,
        opset_version=op_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_batch,
        do_constant_folding=constant_folding
    )
    LOGGER.info("Model Conversion Completed")
    torch_model_stats = os.stat(torch_ckpt)
    onnx_model_stats = os.stat(onnx_ckpt)
    LOGGER.info(f"Torch Model size = {torch_model_stats.st_size}")
    LOGGER.info(f"Onnx Model size = {onnx_model_stats.st_size}")
    ort_session = ort.InferenceSession(onnx_ckpt)
    dummy_input = to_numpy(dummy_input)
    outputs = ort_session.run(
        None,
        {"inputs": dummy_input},
    )
    np.testing.assert_allclose(to_numpy(sample_output), outputs[0], rtol=1e-03, atol=1e-05)
    LOGGER.info("Output of converted model matched")

def main():
    args = get_argument_parser()
    model = PretrainedResNet50()
    if not os.path.exists(os.path.dirname(args.export_path)):
        os.makedirs(os.path.dirname(args.export_path), exist_ok=True)
    LOGGER.info(f"Starting Converting Torch Model at {args.torch_ckpt} into ONNX")
    start_time = time()
    pytorch_to_onnx(model, args.export_path, args.torch_ckpt,
                    constant_folding = args.no_const_folding, dynamic_batch = args.no_dynamic_batch, op_version = args.op_version)    
    elapsed_time = time() - start_time
    LOGGER.info(f"Model successfully converted into ONNX format at {args.export_path}")
    LOGGER.info(f"Model Conversion Time: {elapsed_time}")

if __name__ == "__main__":
    main()

# python3 models/conversion/convert_pytorch_to_onnx.py --torch_ckpt models/pytorch/ckpt/best_resnet50_cifar10.pth --export_path models/onnx/ckpt/best_resnet50_cifar10.onnx
