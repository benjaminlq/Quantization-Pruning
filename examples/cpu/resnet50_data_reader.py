import numpy
import onnxruntime
import os
import os.path as osp
from onnxruntime.quantization import CalibrationDataReader
from PIL import Image


def _preprocess_images(images_folder: str, height: int, width: int, size_limit=0):
    """
    Loads a batch of images and preprocess them
    parameter images_folder: path to folder storing images
    parameter height: image height in pixels
    parameter width: image width in pixels
    parameter size_limit: number of images to load. Default is 0 which means all images are picked.
    return: list of matrices characterizing multiple images
    """
    image_names = os.listdir(images_folder)
    if size_limit > 0 and len(image_names) >= size_limit:
        batch_filenames = [image_names[i] for i in range(size_limit)]
    else:
        batch_filenames = image_names # List of image files
    unconcatenated_batch_data = []

    for image_name in batch_filenames:
        image_filepath = images_folder + "/" + image_name # Path to image file
        pillow_img = Image.new("RGB", (width, height)) # Image File
        pillow_img.paste(Image.open(image_filepath).resize((width, height))) # Image File
        input_data = numpy.float32(pillow_img) - numpy.array(
            [123.68, 116.78, 103.94], dtype=numpy.float32
        ) # Dimension: (heigh, width, channel)
        nhwc_data = numpy.expand_dims(input_data, axis=0) # Dimension = (1, height, width, channel)
        nchw_data = nhwc_data.transpose(0, 3, 1, 2)  # ONNX Runtime standard. Dimension = (bs, channel, height, width)
        unconcatenated_batch_data.append(nchw_data)
    batch_data = numpy.concatenate(
        numpy.expand_dims(unconcatenated_batch_data, axis=0), axis=0
    )
    return batch_data # (num_iters, batch_size, height, width, channel)


class ResNet50DataReader(CalibrationDataReader):
    def __init__(self, calibration_image_folder: str, model_path: str):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, None)
        (_, _, height, width) = session.get_inputs()[0].shape

        # Convert image to input data
        self.nhwc_data_list = _preprocess_images(
            calibration_image_folder, height, width, size_limit=0
        )
        self.input_name = session.get_inputs()[0].name
        self.datasize = len(self.nhwc_data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                [{self.input_name: nhwc_data} for nhwc_data in self.nhwc_data_list]
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None
        
if __name__ == "__main__":
    img_folder = osp.join(osp.dirname(os.path.abspath(__file__)), "test_images")
    model_path = osp.join(osp.dirname(os.path.abspath(__file__)), "mobilenetv2-7.onnx")
    print(f"Path to image folder: {img_folder}")
    session = onnxruntime.InferenceSession(model_path, None)
    (bs, c, height, width) = session.get_inputs()[0].shape
    print(f"ONNX model inputs size: {width} x {height}")
    batch_data = _preprocess_images(img_folder, height = height, width=width)
    print(batch_data.shape)