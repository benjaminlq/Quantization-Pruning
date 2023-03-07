from config import LOGGER
import onnx
import onnxruntime
from data.cifar10 import CIFAR10DataLoader
import argparse
import numpy as np
from time import time

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", "-n", type = str, help = "Experiment Name")
    parser.add_argument("--onnx_ckpt", "-c", type = str, help = "Path to ONNX checkpoint")
    parser.add_argument("--repeats", "-r", type = str, default = 1, help = "Number of evaluation repeats")
    args = parser.parse_args()
    return args

def onnx_eval(ort_session, val_loader, repeats = 1):
    total_no, correct_no = 0, 0
    for images, labels in val_loader:
        for _ in range(repeats):
            images = images.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            ort_inputs = {ort_session.get_inputs()[0].name: images}
            ort_outputs = ort_session.run(None, ort_inputs)
        preds = np.argmax(ort_outputs, axis = 1)
        total_no += preds.shape[0]
        correct_no = (preds == labels).sum()
    
    accuracy = correct_no / total_no * 100
    return accuracy

def main():
    args = get_argument_parser()
    LOGGER.info(f"Test inference on {args.exp_name}")
    LOGGER.info(f"Loading Model from {args.onnx_ckpt}")
    print("Checking ONNX model")
    onnx_model = onnx.load(args.onnx_ckpt)
    onnx.checker.check_model(onnx_model)
    LOGGER.info(f"ONNX model check status OK")
    ort_session = onnxruntime.InferenceSession(args.onnx_ckpt)
    LOGGER.info(f"ONNX checkpoitn loaded succesfully")
    dataloader = CIFAR10DataLoader(batch_size=64)
    val_loader = dataloader.val_dataloader()
    LOGGER.info("Start Onnx Model inference")
    start_time = time()
    onnx_eval(ort_session, val_loader, repeats = args.repeats)
    time_elapsed = time() - start_time
    LOGGER.info(f"Total Inference time: {time_elapsed}")
    
if __name__ == "__main__":
    main()
    