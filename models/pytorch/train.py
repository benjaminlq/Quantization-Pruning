import argparse
import torch
import os
import torch
import torch.nn as nn
from time import time
from config import LOGGER

from models.pytorch.model_resnet import PretrainedResNet50
from models.pytorch.engine import train
from data.cifar10 import CIFAR10DataLoader

default_path = os.path.join(os.path.dirname(__file__), "ckpt", "best_resnet50_cifar10.pth")

def get_argument_parser():
    parser = argparse.ArgumentParser("CIFAR10-Resnet")
    parser.add_argument("--epochs", "-e", type = int, default = 50, help = "Number of Training Epoches")
    parser.add_argument("--batch_size", "-bs", type = int, default = 16, help = "Batch Size")
    parser.add_argument("--learning_rate", "-lr", type = float, default = 1e-4, help = "Initial Learning Rate")
    parser.add_argument("--ckpt_path", "-o", type = str, default = default_path, help = "Check point path for trained model")
    parser.add_argument("--device", "-d", type = str, default = "cpu", help = "Run using CPU or GPU")
    parser.add_argument("--torch_ckpt", "-c", type = str, default = None, help = "Checkpoint for retraining")
    
    args = parser.parse_args()
    return args
    
def main():
    args = get_argument_parser()
    device = torch.device(args.device)
    parent_folder = os.path.dirname(args.ckpt_path)
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder, exist_ok=True)
        
    model = PretrainedResNet50()
    LOGGER.info(f"Training {str(model)} Model using {device}")
    if args.torch_ckpt:
        model.load_state_dict(torch.load(args.torch_ckpt, map_location = device))
        LOGGER.info(f"Model {str(model)} checkpoint loaded succesfully from {args.torch_ckpt}")
        
    model.to(device)
    dataloader = CIFAR10DataLoader(batch_size=args.batch_size)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params = model.parameters(),
                                 lr = args.learning_rate)
    
    LOGGER.info(f"Start training Model for {args.epochs} epochs")
    start_time = time()
    train(model, dataloader, optimizer, criterion, args)
    LOGGER.info(f"Finished training after {time() - start_time}")
    
if __name__ == "__main__":
    main()

# python3 models/pytorch/train.py -bs 16 -e 10 -lr 6e-5 -d cuda -c models/pytorch/ckpt/best_resnet50_cifar10.pth