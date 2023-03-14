import torch
import numpy as np
import time
import onnxruntime as ort
from typing import Optional, Callable

def save_model(model, path):
    torch.save(model.state_dict(), path)
    
def load_model(model, path, device = "cuda"):
    model.load_state_dict(torch.load(path, map_location=device))
    # return model

def to_numpy(tensor: torch.tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def benchmark_inference_time(model_path, model: Optional[Callable] = None, framework: str = "ONNX", runs: int = 10,
                             device = "GPU"):
    if framework.lower() == "onnx":
        if device == "GPU" and ort.get_device() == "GPU":
            session = ort.InferenceSession(model_path,
                                           providers=['CUDAExecutionProvider'])
        else:
            session = ort.InferenceSession(model_path,
                                           providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        
        total_time = 0.0
        input_data = np.zeros((1, 3, 224, 224), dtype = np.float32)
        # Warming Up
        _ = session.run([], {input_name: input_data})
        for i in range(runs):
            start = time.perf_counter()
            _ = session.run([], {input_name: input_data})
            end = (time.perf_counter() - start) * 1000
            total_time += end
            print(f"{end:.2f} ms")
        total_time /= runs
        return f"Avg: {total_time:.2f} ms"
    
    elif framework.lower() == "pytorch":
        if device == "GPU" and torch.cuda.is_available():
            print("Inference benchmark on GPU")
            device = torch.device("cuda")
        else:
            print("Inference benchmark on CPU")
            device = torch.device("cpu")

        load_model(model, model_path, device)
        model.to(device)
        total_time = 0.0
        input_data = torch.zeros((1, 3, 224, 224), device = device, dtype = torch.float32)
        _ = model(input_data)
        for i in range(runs):
            start = time.perf_counter()
            _ = model(input_data)
            end = (time.perf_counter() - start) * 1000
            total_time += end
            print(f"{end:.2f} ms")
        total_time /= runs
        return f"Avg: {total_time:.2f} ms"
    
def generate_aug_model_path(model_path: str) -> str:
    aug_model_path = (
        model_path[: -len(".onnx")] if model_path.endswith(".onnx") else model_path
    )
    return aug_model_path + ".save_tensors.onnx"