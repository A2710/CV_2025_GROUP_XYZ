import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
from tqdm import tqdm
import pandas as pd
import psutil  # For memory usage
import os
from PIL import Image
from torchvision import transforms

# Static batch size as per instruction (choosing 1 for individual query processing)
BATCH_SIZE = 1

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_file_path):
    """Load the TensorRT engine from file."""
    start_time = time.time()
    print(f"Starting to load engine from {engine_file_path} at {time.strftime('%H:%M:%S')}")
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    mem_usage = psutil.virtual_memory().percent
    end_time = time.time()
    print(f"Loaded engine in {end_time - start_time:.4f} seconds, Memory Usage: {mem_usage:.2f}% at {time.strftime('%H:%M:%S')}")
    return engine

def print_io_info(engine):
    """Print the I/O configuration of the engine."""
    start_time = time.time()
    print(f"Starting to print I/O info at {time.strftime('%H:%M:%S')}")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        mode = "Input" if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "Output"
        print(f"{mode} -> Name: {name}, Shape: {shape}, Dtype: {dtype}")
    mem_usage = psutil.virtual_memory().percent
    end_time = time.time()
    print(f"Printed I/O info in {end_time - start_time:.4f} seconds, Memory Usage: {mem_usage:.2f}% at {time.strftime('%H:%M:%S')}")

def allocate_buffers(engine, batch_size=BATCH_SIZE):
    """Allocate GPU and CPU buffers for engine inputs and outputs."""
    start_time = time.time()
    print(f"Starting to allocate buffers with batch_size={batch_size} at {time.strftime('%H:%M:%S')}")
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()

    for i in range(engine.num_io_tensors):
        binding_name = engine.get_tensor_name(i)
        dtype = trt.nptype(engine.get_tensor_dtype(binding_name))
        shape = engine.get_tensor_shape(binding_name)
        shape = tuple(batch_size if dim == -1 else dim for dim in shape)

        size = int(np.prod(shape))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))

        buffer = {
            'name': binding_name,
            'host_mem': host_mem,
            'device_mem': device_mem,
            'shape': shape,
            'dtype': dtype
        }

        if engine.get_tensor_mode(binding_name) == trt.TensorIOMode.INPUT:
            inputs.append(buffer)
        else:
            outputs.append(buffer)

    context = engine.create_execution_context()
    mem_usage = psutil.virtual_memory().percent
    end_time = time.time()
    print(f"Allocated buffers in {end_time - start_time:.4f} seconds, Memory Usage: {mem_usage:.2f}% at {time.strftime('%H:%M:%S')}")
    return inputs, outputs, bindings, stream, context

def run_inference(context, bindings, inputs, outputs, stream, batch_inputs):
    """Run inference using the TensorRT context with CUDA."""
    start_time = time.time()
    print(f"Starting inference for batch of size {len(batch_inputs[0])} at {time.strftime('%H:%M:%S')}")
    try:
        for buffer, data in zip(inputs, batch_inputs):
            np.copyto(buffer['host_mem'], data.ravel())
            cuda.memcpy_htod_async(buffer['device_mem'], buffer['host_mem'], stream)

        context.execute_v2(bindings=bindings)

        for buffer in outputs:
            cuda.memcpy_dtoh_async(buffer['host_mem'], buffer['device_mem'], stream)

        stream.synchronize()
    except Exception as e:
        print(f"Inference failed: {e}")
        return None
    finally:
        end_time = time.time()
        mem_usage = psutil.virtual_memory().percent
        print(f"Completed inference in {end_time - start_time:.4f} seconds, Memory Usage: {mem_usage:.2f}% at {time.strftime('%H:%M:%S')}")

    # Return the single output (e.g., classification or embedding)
    return [buffer['host_mem'].reshape(buffer['shape']) for buffer in outputs]

def preprocess_image(image_path):
    """Preprocess a single image for inference."""
    start_time = time.time()
    print(f"Starting to preprocess image from {image_path} at {time.strftime('%H:%M:%S')}")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])
    img = Image.open(image_path).convert("RGB")
    img_processed = preprocess(img).numpy()[np.newaxis, ...]  # Add batch dimension
    mem_usage = psutil.virtual_memory().percent
    end_time = time.time()
    print(f"Preprocessed image in {end_time - start_time:.4f} seconds, Memory Usage: {mem_usage:.2f}% at {time.strftime('%H:%M:%S')}")
    return img_processed

def run_csv_inference(engine, dataset_dir, csv_path):
    """Run inference for each query from CSV file individually."""
    start_time = time.time()
    print(f"Starting CSV-based inference at {time.strftime('%H:%M:%S')}")
    inputs, outputs, bindings, stream, context = allocate_buffers(engine, batch_size=BATCH_SIZE)

    # Read CSV and process each row
    df = pd.read_csv(csv_path)
    total_entries = len(df)
    print(f"Processing {total_entries} entries from {csv_path}")

    for index, row in tqdm(df.iterrows(), desc="Processing queries", total=total_entries):
        image_path = os.path.join(dataset_dir, row['image_path'].strip())
        # Note: Text is ignored since the engine seems to be image-only

        # Preprocess image
        image_data = preprocess_image(image_path)

        # Run inference (single image input)
        embeddings = run_inference(context, bindings, inputs, outputs, stream, [image_data])
        if embeddings is None:
            print(f"Inference failed for query {index + 1}")
            continue

        # Assuming a single output (e.g., 1000-dimensional vector)
        output = embeddings[0]
        print(f"Query {index + 1}/{total_entries}: Image={image_path}, Output Shape={output.shape}, First Value={output[0, 0]:.4f}")

        mem_usage = psutil.virtual_memory().percent
        print(f"Processed query {index + 1}/{total_entries}, Memory Usage: {mem_usage:.2f}% at {time.strftime('%H:%M:%S')}")

    end_time = time.time()
    mem_usage = psutil.virtual_memory().percent
    print(f"Completed CSV-based inference in {end_time - start_time:.4f} seconds, Memory Usage: {mem_usage:.2f}% at {time.strftime('%H:%M:%S')}")

if __name__ == "__main__":
    start_time = time.time()
    print(f"Starting inference script at {time.strftime('%H:%M:%S')}")
    engine_path = "/media/seas_au/prsdata/xyz/cv_final/ICFG-PDES/ICFG-PEDES/final_modelint8.engine"
    dataset_dir = "/media/seas_au/prsdata/xyz/cv_final/ICFG-PDES/ICFG-PEDES"
    csv_path = "/media/seas_au/prsdata/xyz/cv_final/ICFG-PDES/ICFG-PEDES/temp_first_100.csv"

    engine = load_engine(engine_path)
    print("\nEngine Info:")
    print_io_info(engine)

    run_csv_inference(engine, dataset_dir, csv_path)

    end_time = time.time()
    mem_usage = psutil.virtual_memory().percent
    print(f"Completed inference script in {end_time - start_time:.4f} seconds, Memory Usage: {mem_usage:.2f}% at {time.strftime('%H:%M:%S')}")

    # Note: Hardware acceleration with NVJPG, DLA, NVENC, NVDEC SE, VIC can be implemented
    # by configuring TensorRT with appropriate plugins. This requires additional setup
    # and is not directly coded here but can be enabled in the TensorRT context.
