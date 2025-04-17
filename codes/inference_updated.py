import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
from typing import List, Tuple, Optional
import pandas as pd
from PIL import Image
from torchvision import transforms

# Logger for TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.INFO)

class MyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_data, batch_size=32, input_shape=None):
        super().__init__()
        self.data = calibration_data
        self.batch_size = batch_size
        self.current_index = 0

        # If input shape is provided, use it to calculate nbytes
        if input_shape is not None:
            element_size = np.dtype(np.float32).itemsize  # Assuming float32 input
            total_size = batch_size * np.prod(input_shape) * element_size
            self.device_input = cuda.mem_alloc(int(total_size))
        else:
            # Otherwise use the first batch's size
            self.device_input = cuda.mem_alloc(self.batch_size * self.data[0].nbytes)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.data):
            return None

        batch = self.data[self.current_index:self.current_index + self.batch_size]
        batch = np.ascontiguousarray(batch)
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size

        return [int(self.device_input)]

    def read_calibration_cache(self):
        cache_file = "/media/seas_au/prsdata/xyz/cv_final/ICFG-PDES/ICFG-PEDES/calib.cache"
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open("calibration.cache", "wb") as f:
            f.write(cache)

def prepare_calibration_data(dataset_path: str, batch_size: int, input_shape: Tuple, num_batches: int = 10) -> np.ndarray:
    """
    Prepare calibration data from the ICFG-PEDES dataset.

    Args:
        dataset_path: Path to the ICFG-PEDES dataset
        batch_size: Size of each batch
        input_shape: Shape of a single input (e.g., (3, 224, 224))
        num_batches: Number of batches to use for calibration

    Returns:
        Numpy array of preprocessed calibration data
    """
    csv_path = "/media/seas_au/prsdata/xyz/cv_final/ICFG-PDES/ICFG-PEDES/temp.csv"
    df = pd.read_csv(csv_path)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                           std=[0.26862954, 0.26130258, 0.27577711])
    ])

    total_samples = batch_size * num_batches
    calibration_data = []

    for index, row in df.iterrows():
        if len(calibration_data) >= total_samples:
            break

        img_path = f"{dataset_path}/{row['image_path']}"
        try:
            img = Image.open(img_path).convert("RGB")
            img_processed = preprocess(img).numpy()
            calibration_data.append(img_processed)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            continue

    if not calibration_data:
        raise ValueError("No valid images processed for calibration")

    calibration_data = np.array(calibration_data, dtype=np.float32)
    print(f"Prepared {len(calibration_data)} samples for calibration")
    return calibration_data

def build_engine(
    onnx_file_path: str,
    engine_file_path: str,
    precision: str = "fp32",
    calibration_data: Optional[np.ndarray] = None,
    batch_size: int = 1,
    workspace_size: int = 1 << 30
) -> trt.ICudaEngine:
    """
    Build a TensorRT engine from an ONNX file.

    Args:
        onnx_file_path: Path to the ONNX model
        engine_file_path: Path where the TensorRT engine will be saved
        precision: Precision mode - 'fp32', 'fp16', or 'int8'
        calibration_data: Data for INT8 calibration
        batch_size: Maximum batch size for the engine
        workspace_size: Maximum workspace size for TensorRT

    Returns:
        TensorRT engine
    """
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()
        config.max_workspace_size = workspace_size

        if precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("Using FP16 precision")
        elif precision == "int8" and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("Using INT8 precision")
            if calibration_data is None:
                raise ValueError("Calibration data is required for INT8 precision")
            input_shape = calibration_data[0].shape
            calibrator = MyCalibrator(calibration_data, batch_size=min(batch_size, len(calibration_data)), input_shape=input_shape)
            config.int8_calibrator = calibrator
            print(f"INT8 calibration set up with {len(calibration_data)} samples")
        else:
            print(f"Using FP32 precision (requested {precision})")

        print(f"Parsing ONNX file: {onnx_file_path}")
        with open(onnx_file_path, "rb") as f:
            if not parser.parse(f.read()):
                print("ERROR: Failed to parse the ONNX file")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # Set optimization profile for dynamic batch sizes
        profile = builder.create_optimization_profile()
        # Assuming input name is 'input' (replace with actual name from ONNX model)
        profile.set_shape("input", (1, 3, 224, 224), (batch_size, 3, 224, 224), (batch_size, 3, 224, 224))
        config.add_optimization_profile(profile)

        print("Building TensorRT engine...")
        start_time = time.time()
        engine = builder.build_engine(network, config)
        build_time = time.time() - start_time
        print(f"Engine built in {build_time:.2f} seconds")

        if engine is None:
            print("ERROR: Failed to create the TensorRT engine")
            return None

        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())
            print(f"Engine serialized to: {engine_file_path}")

        return engine

def load_engine(engine_file_path: str) -> trt.ICudaEngine:
    """
    Load a TensorRT engine from file.

    Args:
        engine_file_path: Path to the serialized engine file

    Returns:
        TensorRT engine
    """
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def run_inference(engine: trt.ICudaEngine, input_data: np.ndarray) -> List[np.ndarray]:
    """
    Run inference using a TensorRT engine.

    Args:
        engine: TensorRT engine
        input_data: Input data for inference

    Returns:
        List of output tensors
    """
    with engine.create_execution_context() as context:
        # Print I/O info (similar to previous code)
        print("Engine I/O Configuration:")
        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            shape = context.get_binding_shape(i)
            dtype = trt.nptype(engine.get_binding_dtype(i))
            mode = "Input" if engine.binding_is_input(i) else "Output"
            print(f"{mode} -> Name: {name}, Shape: {shape}, Dtype: {dtype}")

        # Allocate device memory
        d_input = cuda.mem_alloc(input_data.nbytes)
        output_bindings = []
        outputs = []

        for i in range(engine.num_bindings):
            if engine.binding_is_input(i):
                continue
            output_shape = context.get_binding_shape(i)
            output_size = trt.volume(output_shape)
            output_dtype = trt.nptype(engine.get_binding_dtype(i))
            d_output = cuda.mem_alloc(int(output_size * output_dtype.itemsize))
            output_bindings.append(d_output)
            outputs.append(np.empty(output_shape, dtype=output_dtype))

        bindings = [int(d_input)] + [int(d_output) for d_output in output_bindings]

        # Copy input to device
        cuda.memcpy_htod(d_input, input_data)

        # Run inference
        context.execute_v2(bindings)

        # Copy outputs to host
        for i, d_output in enumerate(output_bindings):
            cuda.memcpy_dtoh(outputs[i], d_output)

        return outputs

def main():
    # Define paths and parameters
    onnx_model_path = "/media/seas_au/prsdata/xyz/cv_final/ICFG-PDES/ICFG-PEDES/final_model.onnx"  # Update with your actual ONNX path
    engine_path = "/media/seas_au/prsdata/xyz/cv_final/ICFG-PDES/ICFG-PEDES/final_modelint8.engine"
    input_shape = (3, 224, 224)
    batch_size = 8
    precision = "int8"

    # Prepare calibration data
    if precision == "int8":
        calibration_data = prepare_calibration_data(
            dataset_path="/media/seas_au/prsdata/xyz/cv_final/ICFG-PDES/ICFG-PEDES",
            batch_size=batch_size,
            input_shape=input_shape,
            num_batches=10
        )
    else:
        calibration_data = None

    # Build the TensorRT engine
    engine = build_engine(
        onnx_file_path=onnx_model_path,
        engine_file_path=engine_path,
        precision=precision,
        calibration_data=calibration_data,
        batch_size=batch_size
    )

    # Run inference with a sample input
    if engine:
        sample_input = calibration_data[:1] if calibration_data is not None else np.random.random((1,) + input_shape).astype(np.float32)
        outputs = run_inference(engine, sample_input)
        print(f"Inference complete. Number of outputs: {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"Output {i} shape: {output.shape}")

if __name__ == "__main__":
    main()
