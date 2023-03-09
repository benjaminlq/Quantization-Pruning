from config import LOGGER
import onnx
import torch ### Need to import torch before onnxruntime to activate cuDNN/CUDA for CUDAExecutionProvider
import onnxruntime as ort
from data.cifar10 import CIFAR10DataLoader
import argparse
import numpy as np
from time import time
from tqdm import tqdm

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", "-n", type = str, help = "Experiment Name")
    parser.add_argument("--onnx_ckpt", "-c", type = str, help = "Path to ONNX checkpoint")
    parser.add_argument("--repeats", "-r", type = int, default = 1, help = "Number of evaluation repeats")
    parser.add_argument("--device", "-d", type = str, default = "GPU", help = "GPU or CPU")
    args = parser.parse_args()
    return args

def onnx_eval(ort_session, val_loader, repeats = 1):
    total_no, correct_no = 0, 0
    tk0 = tqdm(val_loader, total=len(val_loader))
    for images, labels in tk0:
        images = images.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        for _ in range(repeats):
            ort_inputs = {ort_session.get_inputs()[0].name: images}
            ort_outputs = ort_session.run(None, ort_inputs)
        preds = np.argmax(ort_outputs[0], axis = 1)
        total_no += preds.shape[0]
        correct_no += (preds == labels).sum()
    
    accuracy = correct_no * 100 / total_no
    return accuracy

def main():
    args = get_argument_parser()
    LOGGER.info(f"Test inference on {args.exp_name}")
    LOGGER.info(f"Loading Model from {args.onnx_ckpt}")
    print("Checking ONNX model")
    onnx_model = onnx.load(args.onnx_ckpt)
    onnx.checker.check_model(onnx_model)
    LOGGER.info(f"ONNX model check status OK")
    device = ort.get_device()
    LOGGER.info(f"ONNX inference with {args.device}")
    if args.device == "GPU":
        assert device == args.device
        ort_session = ort.InferenceSession(args.onnx_ckpt,
                                           providers=['CUDAExecutionProvider'])
    else:
        ort_session = ort.InferenceSession(args.onnx_ckpt,
                                           providers=['CPUExecutionProvider'])
        
    LOGGER.info(f"ONNX checkpoint loaded successfully")
    dataloader = CIFAR10DataLoader(batch_size=16)
    val_loader = dataloader.val_dataloader()
    LOGGER.info("Start Onnx Model inference")
    start_time = time()
    accuracy = onnx_eval(ort_session, val_loader, repeats = args.repeats)
    time_elapsed = time() - start_time
    LOGGER.info(f"ONNX Model Accuracy: {accuracy}")
    LOGGER.info(f"Total Inference time: {time_elapsed}")
    
if __name__ == "__main__":
    main()
    
# python3 models/onnx/cifar_inference.py -n "ResNet50 CIFAR10 ONNX Model" -c models/onnx/ckpt/best_resnet50_cifar10.onnx -r 1 -d GPU