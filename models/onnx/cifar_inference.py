import onnxruntime

def onnx_eval(ort_session, val_loader):
    total_no, correct_no = 0, 0
    for images, labels in val_loader:
        images = images.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
                

def main():
    args = get_argument_parser()
    ort_session = onnxruntime(args.onnx_ckpt)
    dataloader = 