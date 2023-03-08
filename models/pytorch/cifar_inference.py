from config import LOGGER
from data.cifar10 import CIFAR10DataLoader
import argparse
import numpy as np
from time import time
import torch
from models.pytorch.model_resnet import PretrainedResNet50
from utils import load_model

def get_argument_parser():
    parser = argparse.ArgumentParser("Pytorch Inference")
    parser.add_argument("--exp_name", "-n", type = str, help = "Experiment Name")
    parser.add_argument("--torch_ckpt", "-c", type = str, help = "Path to ONNX checkpoint")
    parser.add_argument("--repeats", "-r", type = int, default = 1, help = "Number of evaluation repeats")
    parser.add_argument("--device", "-d", type = str, default = "cuda", help = "Use CPU or GPU for inference")
    args = parser.parse_args()
    return args

def pytorch_eval(torch_model, val_loader, device, repeats = 1):
    total_no, correct_no = 0, 0
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        for _ in range(repeats):
            outs = torch_model(images)
        _, preds = torch.max(outs, 1)
        total_no += preds.size(0)
        correct_no += (preds == labels).sum().item()
    
    accuracy = correct_no * 100 / total_no 
    return accuracy

def main():
    args = get_argument_parser()
    LOGGER.info(f"Test inference on {args.exp_name}")
    LOGGER.info(f"Loading Model from {args.torch_ckpt}")
    device = torch.device(args.device)
    torch_model = PretrainedResNet50()
    torch_model.to(device)
    load_model(torch_model, args.torch_ckpt, args.device)
    LOGGER.info(f"Model checkpoint loaded successfully from {args.torch_ckpt}")
    
    dataloader = CIFAR10DataLoader(batch_size=16)
    val_loader = dataloader.val_dataloader()
    
    LOGGER.info("Start Pytorch Model inference")
    start_time = time()
    accuracy = pytorch_eval(torch_model, val_loader, device, repeats = args.repeats)
    time_elapsed = time() - start_time
    LOGGER.info(f"Pytorch Model Accuracy: {accuracy}")
    LOGGER.info(f"Total Inference time: {time_elapsed}")
    
if __name__ == "__main__":
    main()
    
# python3 models/pytorch/cifar_inference.py -n "ResNet50 CIFAR10 Pytorch Model" -c models/pytorch/ckpt/best_resnet50_cifar10.pth -r 1