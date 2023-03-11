import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader
from data.cifar10 import CIFAR10DataLoader

class CIFAR10DataReader(CalibrationDataReader):
    def __init__(
        self,
        model_path: str,
        batch_size: int = 8,
        batch_no: int = 10,
    ):
        self.enum_data = None
        if ort.get_device() == "GPU":
            session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])
        else:
            session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        (_, _, height, width) = session.get_inputs()[0].shape
        datamanager = CIFAR10DataLoader(batch_size=batch_size,
                                        input_size = (3, height, width))
        self.datasize = batch_no
        self.input_name = session.get_inputs()[0].name
        self.data_list = []
        train_iter = iter(datamanager.train_dataloader())
        for _ in range(batch_no):
            imgs, _ = next(train_iter)
            batch_data = imgs.detach().cpu().numpy()
            self.data_list.append(batch_data)
        
    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: batch_data for batch_data in self.data_list}]
            )
        return next(self.enum_data, None)
        
    def rewind(self):
       self.enum_data = None 
        
        