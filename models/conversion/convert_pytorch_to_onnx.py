# def check_converted_model(
#     torch_model: str,
#     onnx_model: str,
#     tolerance: float
# ):
#     assert

import torch
import torchvision
from typing import Tuple, List
import os
import argparse

def get_argument_parser():
    parser = argparse.ArgumentParser("Torch-to-ONNX conversion")
    parser.add_argument("--torch_ckpt", "-t", type = str, help = "Path to Torch Model")
    parser.add_argument("--export_path", "-x", type = str, help = "Export Path to store Onnx Model")
    args = parser.parse_args()
    return args
    
def pytorch_to_onnx(
    torch_ckpt,
    onnx_ckpt,
    input_size: Tuple = (3, 224, 224),
    input_names: List = ["inputs"],
    output_names: List = ["outputs"],
    dynamic_batch: bool = True,
):
    model = torchvision.models.resnet50()
    print(f"Loading Model from Checkpoint {torch_ckpt}.")
    model.load_state_dict(torch.load(torch_ckpt))
    print(f"Finished loading model.")
    dummy_input = torch.randn(1, *input_size, requires_grad=True)
    model.eval()
    dynamic_batch = {"inputs": {0: "batch_size"},
                     "outputs": {0: "batch_size"}} if dynamic_batch else None
    print("Start Converting Pytorch Model to Onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_ckpt,
        export_params=True,
        opset_version=11,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_batch
    )
    print("Model Conversion Completed")
    torch_model_stats = os.stat(torch_ckpt)
    onnx_model_stats = os.stat(onnx_ckpt)
    print(f"Torch Model size = {torch_model_stats.st_size}")
    print(f"Onnx Model size = {onnx_model_stats.st_size}")

if __name__ == "__main__":
    args = get_argument_parser()
    
    torch_ckpt = "onnx/artifacts/resnet50-0676ba61.pth"
    onnx_ckpt = "onnx/artifacts/onnx_resnet50.onnx"
    pytorch_to_onnx(torch_ckpt, onnx_ckpt)

