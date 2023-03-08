import argparse
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

def get_argument_parser():
    parser = argparse.ArgumentParser("ONNX Quantization")
    parser.add_argument("--onnx_ckpt", "-c", type = str, help = "Path to Float32 ONNX Model")
    parser.add_argument("--export_path", "-e", type = str, help = "Path to Quantized ONNX Model")
    parser.add_argument("--calibrate_datatset", "-d", type = str, help = "Path to Calibration Dataset")
    parser.add_argument("--quant_format", "-f", type = QuantFormat.from_string, default = QuantFormat.QDQ, help = "Quantization Format")
    parser.add_argument("--per_channel", action = "store_true", help = "To be added")
    
    args = parser.parse_args()
    return args

def main():
    args = get_argument_parser()
    
if __name__ =="__main__":
    main()
    
# python3 quantize/onnx_quantize.py 