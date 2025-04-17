import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
from tqdm import tqdm
import pandas as pd
import psutil
import os
from PIL import Image
from torchvision import transforms
import cupy as cp  # Import CuPy for GPU operations
import csv
import pynvml  # For GPU utilization (optional, install with `pip install pynvml`)

# Static batch size as per instruction (choosing 4 for batch processing)
BATCH_SIZE = 4

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def log_memory_time(message, start_time=None):
    """Log memory usage and elapsed time if start_time is provided."""
    mem_usage = psutil.virtual_memory().percent
    if start_time:
        elapsed = time.time() - start_time
        print(f"{message} in {elapsed:.4f} seconds, Memory Usage: {mem_usage:.2f}% at {time.strftime('%H:%M:%S')}")
    else:
        print(f"{message}, Memory Usage: {mem_usage:.2f}% at {time.strftime('%H:%M:%S')}")

def load_engine(engine_file_path, verbose=False):
    """Load TensorRT engine and optionally print I/O info."""
    start_time = time.time()
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    if verbose:
        print("\nEngine Info:")
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            mode = "Input" if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "Output"
            print(f"{mode} -> Name: {name}, Shape: {shape}, Dtype: {dtype}")

    log_memory_time(f"Loaded engine from {engine_file_path}", start_time)
    return engine

def allocate_buffers(engine, batch_size=BATCH_SIZE):
    """Allocate buffers and set input shapes."""
    start_time = time.time()
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    context = engine.create_execution_context()

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        shape = engine.get_tensor_shape(name)
        shape = tuple(batch_size if dim == -1 else dim for dim in shape)

        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            context.set_input_shape(name, shape)

        size = int(np.prod(shape))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))

        buffer = {'name': name, 'host_mem': host_mem, 'device_mem': device_mem, 'shape': shape}
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append(buffer)
        else:
            outputs.append(buffer)

    log_memory_time(f"Allocated buffers with batch_size={batch_size}", start_time)
    return inputs, outputs, bindings, stream, context

def run_inference(context, bindings, inputs, outputs, stream, batch_inputs):
    """Run inference with error handling."""
    start_time = time.time()
    try:
        for buffer, data in zip(inputs, batch_inputs):
            np.copyto(buffer['host_mem'], data.ravel())
            cuda.memcpy_htod_async(buffer['device_mem'], buffer['host_mem'], stream)

        context.execute_v2(bindings=bindings)
        for buffer in outputs:
            cuda.memcpy_dtoh_async(buffer['host_mem'], buffer['device_mem'], stream)
        stream.synchronize()

        return outputs[0]['host_mem']
    except Exception as e:
        print(f"Inference failed: {str(e)}")
        return None
    finally:
        log_memory_time(f"Completed inference for batch of size {len(batch_inputs[0]) if batch_inputs else 0}", start_time)

def preprocess_image(image_path):
    """Preprocess a single image."""
    start_time = time.time()
    try:
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711])
        ])
        img = Image.open(image_path).convert("RGB")
        img_processed = preprocess(img).numpy()[np.newaxis, ...]  # Shape: [1, 3, 224, 224]
        log_memory_time(f"Preprocessed image from {image_path}", start_time)
        return img_processed
    except Exception as e:
        print(f"Failed to preprocess {image_path}: {str(e)}")
        return None

def evaluate_recall_mAP(image_embeddings, text_embeddings, labels, k_list=[1, 5, 10]):
    """Evaluate Recall@k and mAP for image-to-text retrieval on GPU using CuPy."""
    start_time = time.time()
    # Convert inputs to CuPy arrays (move to GPU)
    image_embeddings = cp.array(image_embeddings)
    text_embeddings = cp.array(text_embeddings)
    labels = cp.array(labels)
    n_samples = image_embeddings.shape[0]
    recall_at_k = {k: 0 for k in k_list}
    avg_precision_list = []

    for i in tqdm(range(n_samples), desc="Evaluating"):
        # Compute cosine similarity on GPU
        img_emb = image_embeddings[i:i+1]
        norm_img = cp.sqrt(cp.sum(img_emb ** 2, axis=1, keepdims=True))
        norm_text = cp.sqrt(cp.sum(text_embeddings ** 2, axis=1, keepdims=True))
        sim_scores = cp.dot(img_emb, text_embeddings.T) / (norm_img * norm_text.T)
        sim_scores = sim_scores.ravel()

        # Sort indices by similarity (descending)
        sorted_indices = cp.argsort(-sim_scores)
        gt_label = labels[i]

        # Compute Recall@k
        for k in k_list:
            top_k_labels = labels[sorted_indices[:k]]
            if gt_label in top_k_labels:
                recall_at_k[k] += 1

        # Compute Average Precision
        precisions = []
        relevant_found = 0
        for rank, idx in enumerate(sorted_indices, 1):
            if labels[idx] == gt_label:
                relevant_found += 1
                precisions.append(relevant_found / rank)
                break
        avg_precision = cp.mean(cp.array(precisions)) if precisions else 0
        avg_precision_list.append(float(avg_precision))  # Convert to Python float for final aggregation

    # Compute final metrics on CPU for simplicity
    total = n_samples
    recall_metrics = {k: recall_at_k[k] / total for k in k_list}
    mAP = np.mean(avg_precision_list)
    print("\n🔹 Evaluation Metrics:")
    for k in k_list:
        print(f"  - Recall@{k}: {recall_metrics[k]:.4f}")
    print(f"  - Mean Average Precision (mAP): {mAP:.4f}")
    log_memory_time("Completed GPU-based evaluation", start_time)
    return recall_metrics, mAP

def run_csv_inference_and_evaluate(engine, dataset_dir, csv_path, summary_file_name="inference_summary.log.csv"):
    """Run inference and evaluate with summary metrics using batch processing."""
    start_time = time.time()
    inputs, outputs, bindings, stream, context = allocate_buffers(engine)

    df = pd.read_csv(csv_path)
    if 'image_path' not in df.columns:
        raise ValueError("CSV must contain 'image_path' column")
    total_entries = len(df)
    print(f"Processing {total_entries} entries from {csv_path}")

    embeddings = []
    labels = []
    inference_times = []
    gpu_usages = []  # To store GPU usage per query
    gpu_handle = initialize_gpu()
    total_inference_time = 0.0
    peak_memory = 0.0
    peak_cpu = 0.0
    peak_gpu = 0.0
    error_count = 0

    # Buffer for batch processing
    batch_images = []
    batch_labels = []
    processed_indices = []

    for index, row in tqdm(enumerate(df.iterrows()), desc="Processing queries", total=total_entries):
        img_idx, row = row
        image_path = os.path.join(dataset_dir, row['image_path'].strip())
        label = row.get('caption_id', img_idx)

        image_data = preprocess_image(image_path)
        if image_data is None:
            error_count += 1
            continue

        batch_images.append(image_data)
        batch_labels.append(label)
        processed_indices.append(img_idx)

        # Process batch when reaching BATCH_SIZE or at the end
        if len(batch_images) == BATCH_SIZE or index == total_entries - 1:
            if batch_images:  # Ensure there are images to process
                # Stack images into a batch (shape: [BATCH_SIZE, 3, 224, 224])
                batch_data = np.zeros((BATCH_SIZE, 3, 224, 224), dtype=np.float32)
                actual_batch_size = len(batch_images)
                for i, img in enumerate(batch_images):
                    batch_data[i] = img[0]  # Extract [1, 3, 224, 224] to [3, 224, 224] and place in batch

                inference_start = time.time()
                embedding = run_inference(context, bindings, inputs, outputs, stream, [batch_data])
                inference_end = time.time()
                inference_time = inference_end - inference_start
                inference_times.append(inference_time)
                total_inference_time += inference_time

                cpu_usage = psutil.cpu_percent()
                memory_usage = psutil.virtual_memory().percent
                gpu_usage = get_gpu_usage(gpu_handle) if gpu_handle else 0.0
                gpu_usages.append(gpu_usage)  # Record GPU usage

                peak_memory = max(peak_memory, memory_usage)
                peak_cpu = max(peak_cpu, cpu_usage)
                peak_gpu = max(peak_gpu, gpu_usage)

                if embedding is None:
                    print(f"Batch inference failed for indices {processed_indices}")
                    error_count += actual_batch_size
                else:
                    # Split batch embeddings (assuming output shape is [BATCH_SIZE, 1000])
                    batch_embeddings = embedding.reshape(actual_batch_size, -1)
                    for i in range(actual_batch_size):
                        embeddings.append(batch_embeddings[i])
                        labels.append(batch_labels[i])
                        print(f"Query {processed_indices[i] + 1}/{total_entries}: Image={batch_images[i].shape}, Output Shape={batch_embeddings[i].shape}, First Value={batch_embeddings[i][0]:.4f}")

                # Clear batch for next iteration
                batch_images = []
                batch_labels = []
                processed_indices = []

    if embeddings:
        embeddings = np.array(embeddings)
        labels = np.array(labels)
        # Placeholder: Assume text_embeddings are same for simplicity
        # Replace with actual text embeddings loading logic
        recall_metrics, mAP = evaluate_recall_mAP(embeddings, embeddings, labels)
    else:
        print("No valid embeddings generated; skipping evaluation.")
        recall_metrics = {k: 0.0 for k in [1, 5, 10]}
        mAP = 0.0

    # Calculate summary metrics
    total_time = time.time() - start_time
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    avg_cpu = np.mean([psutil.cpu_percent(interval=1) for _ in range(5)]) if total_entries > 0 else 0  # Approximate average
    avg_memory = np.mean([psutil.virtual_memory().percent for _ in range(5)]) if total_entries > 0 else 0  # Approximate average
    avg_gpu = np.mean(gpu_usages) if gpu_usages and total_entries > 0 else 0  # Average GPU usage from per-query measurements
    avg_qps = total_entries / total_time if total_time > 0 else 0

    # Write summary to CSV with custom filename
    with open(summary_file_name, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Total_Queries", "Total_Time_sec", "Total_Inference_Time_sec", "Avg_Inference_Time_sec",
            "Avg_QPS", "Avg_CPU_Usage_%", "Avg_Memory_Usage_%", "Avg_GPU_Usage_%",
            "Peak_CPU_Usage_%", "Peak_Memory_Usage_%", "Peak_GPU_Usage_%", "Error_Count",
            "Recall@1", "Recall@5", "Recall@10", "Mean_Average_Precision_mAP", "Timestamp"
        ])
        writer.writerow([
            total_entries, f"{total_time:.4f}", f"{total_inference_time:.4f}", f"{avg_inference_time:.4f}",
            f"{avg_qps:.2f}", f"{avg_cpu:.2f}", f"{avg_memory:.2f}", f"{avg_gpu:.2f}",
            f"{peak_cpu:.2f}", f"{peak_memory:.2f}", f"{peak_gpu:.2f}", error_count,
            f"{recall_metrics[1]:.4f}", f"{recall_metrics[5]:.4f}", f"{recall_metrics[10]:.4f}", f"{mAP:.4f}",
            time.strftime('%H:%M:%S')
        ])

    print(f"\n🔹 Summary Metrics:")
    print(f"  - Total Queries: {total_entries}")
    print(f"  - Total Time: {total_time:.4f} seconds")
    print(f"  - Total Inference Time: {total_inference_time:.4f} seconds")
    print(f"  - Average Inference Time per Query: {avg_inference_time:.4f} seconds")
    print(f"  - Average Queries per Second (QPS): {avg_qps:.2f}")
    print(f"  - Average CPU Usage: {avg_cpu:.2f}%")
    print(f"  - Average Memory Usage: {avg_memory:.2f}%")
    print(f"  - Average GPU Usage: {avg_gpu:.2f}%")
    print(f"  - Peak CPU Usage: {peak_cpu:.2f}%")
    print(f"  - Peak Memory Usage: {peak_memory:.2f}%")
    print(f"  - Peak GPU Usage: {peak_gpu:.2f}%")
    print(f"  - Error Count: {error_count}")
    print(f"  - Recall@1: {recall_metrics[1]:.4f}")
    print(f"  - Recall@5: {recall_metrics[5]:.4f}")
    print(f"  - Recall@10: {recall_metrics[10]:.4f}")
    print(f"  - Mean Average Precision (mAP): {mAP:.4f}")

    log_memory_time("Completed inference and evaluation", start_time)

def initialize_gpu():
    """Initialize NVIDIA management library for GPU usage."""
    try:
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming first GPU
    except Exception as e:
        print(f"Warning: Could not initialize GPU monitoring: {e}. GPU usage will be 0.0%")
        return None

def get_gpu_usage(handle):
    """Get GPU utilization percentage."""
    if handle is None:
        return 0.0
    try:
        return pynvml.nmlDeviceGetUtilizationRates(handle).gpu
    except Exception:
        return 0.0

if __name__ == "__main__":
    start_time = time.time()
    engine_path = "/media/seas_au/prsdata/xyz/cv_final/modelfp32.engine"
    dataset_dir = "/media/seas_au/prsdata/xyz/cv_final/ICFG-PDES/ICFG-PEDES"
    csv_path = "/media/seas_au/prsdata/xyz/cv_final/ICFG-PDES/ICFG-PEDES/temp_first_1000.csv"
    summary_file_name = "results_fp32_batchsize4.csv"  # Example custom name

    engine = load_engine(engine_path, verbose=True)
    run_csv_inference_and_evaluate(engine, dataset_dir, csv_path, summary_file_name)
    log_memory_time("Completed script", start_time)

    # Note: Hardware acceleration with NVJPG, DLA, NVENC, NVDEC SE, VIC can be implemented
    # by configuring TensorRT with appropriate plugins. This requires additional setup
    # and is not directly coded here but can be enabled in the TensorRT context.
