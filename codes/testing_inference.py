import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import clip
import pandas as pd
import psutil
import pynvml
import logging
import os
from datetime import datetime
import cvcuda
import cupy as cp
import h5py
import gc

# Initialize logging
log_dir = "/media/seas_au/prsdata/xyz/cv_final/ICFG-PDES/ICFG-PEDES/logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file)]
)
logger = logging.getLogger(__name__)

# Initialize NVIDIA Management Library for GPU monitoring
try:
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    if device_count > 0:
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    else:
        logger.warning("No NVIDIA GPU found. GPU metrics will be skipped.")
        gpu_handle = None
except pynvml.NVMLError as e:
    logger.warning(f"Failed to initialize pynvml: {e}. GPU metrics will be skipped.")
    gpu_handle = None

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_file_path):
    """Load the TensorRT engine from file."""
    logger.info(f"Loading TensorRT engine from {engine_file_path}")
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        logger.info("TensorRT engine loaded successfully")
        return engine

def print_io_info(engine):
    """Print the I/O configuration of the engine."""
    logger.info("I/O Configuration:")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        mode = "Input" if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "Output"
        logger.info(f"{mode} -> Name: {name}, Shape: {shape}, Dtype: {dtype}")

def get_system_metrics():
    """Retrieve CPU and GPU memory usage, core usage, and GPU power."""
    metrics = {}
    process = psutil.Process()
    cpu_mem = process.memory_info().rss / (1024 ** 2)
    metrics["cpu_memory_mb"] = cpu_mem
    metrics["cpu_cores_total"] = psutil.cpu_count()
    try:
        metrics["cpu_cores_used"] = len(process.cpu_affinity())
    except psutil.Error:
        metrics["cpu_cores_used"] = "Unknown"

    if gpu_handle:
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            metrics["gpu_memory_used_mb"] = mem_info.used / (1024 ** 2)
            metrics["gpu_memory_total_mb"] = mem_info.total / (1024 ** 2)
            metrics["gpu_power_watts"] = pynvml.nvmlDeviceGetPowerUsage(gpu_handle) / 1000.0
            metrics["gpu_name"] = pynvml.nvmlDeviceGetName(gpu_handle)
            metrics["gpu_utilization_percent"] = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
        except pynvml.NVMLError as e:
            logger.warning(f"Failed to retrieve GPU metrics: {e}")
            metrics.update({
                "gpu_memory_used_mb": "Unknown",
                "gpu_memory_total_mb": "Unknown",
                "gpu_power_watts": "Unknown",
                "gpu_name": "Unknown",
                "gpu_utilization_percent": "Unknown"
            })
    else:
        metrics.update({
            "gpu_memory_used_mb": "N/A",
            "gpu_memory_total_mb": "N/A",
            "gpu_power_watts": "N/A",
            "gpu_name": "N/A",
            "gpu_utilization_percent": "N/A"
        })

    return metrics

def allocate_buffers(engine, batch_size=1):
    """Allocate GPU and CPU buffers for engine inputs and outputs."""
    logger.info(f"Allocating buffers for batch_size={batch_size}")
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    device_mems = []

    cpu_memory_before = psutil.Process().memory_info().rss / (1024 ** 2)
    gpu_memory_before = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / (1024 ** 2) if gpu_handle else 0

    for i in range(engine.num_io_tensors):
        binding_name = engine.get_tensor_name(i)
        dtype = trt.nptype(engine.get_tensor_dtype(binding_name))
        shape = engine.get_tensor_shape(binding_name)
        shape = tuple(batch_size if dim == -1 else dim for dim in shape)

        size = int(np.prod(shape))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        device_mems.append(device_mem)

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

    cpu_memory_after = psutil.Process().memory_info().rss / (1024 ** 2)
    gpu_memory_after = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used / (1024 ** 2) if gpu_handle else 0
    logger.info(f"CPU memory allocated: {(cpu_memory_after - cpu_memory_before):.2f} MB")
    logger.info(f"GPU memory allocated: {(gpu_memory_after - gpu_memory_before) if gpu_memory_after else 'Unknown'} MB")

    def cleanup():
        for dm in device_mems:
            dm.free()
        stream.synchronize()

    return inputs, outputs, bindings, stream, context, cleanup

def run_inference(context, bindings, inputs, outputs, stream, batch_inputs):
    """Run inference using TensorRT with CVCUDA preprocessing."""
    logger.info(f"Running inference for batch of size {len(batch_inputs[0])}")
    metrics_before = get_system_metrics()

    try:
        start_time = time.time()
        cvcuda_stream = cvcuda.Stream()

        # Convert batch_inputs to CVCUDA tensor
        batch_tensor = torch.from_numpy(batch_inputs[0]).to("cuda").to(torch.float16)
        nv_tensor = cvcuda.as_tensor(batch_tensor, "NHWC")

        # Preprocess with CVCUDA
        with cvcuda_stream:
            nv_resized = cvcuda.resize(nv_tensor, (len(batch_inputs[0]), 224, 224, 3), cvcuda.Interp.CUBIC)
            nv_float = cvcuda.convertto(nv_resized, np.float16, scale=1/255)
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
            scale_tensor = torch.tensor([1/std[0], 1/std[1], 1/std[2]], device="cuda").reshape(1, 1, 1, 3)
            base_tensor = torch.tensor(mean, device="cuda").reshape(1, 1, 1, 3)
            nv_processed = cvcuda.normalize(
                nv_float, base_tensor, scale_tensor, cvcuda.NormalizeFlags.SCALE_IS_STDDEV
            )

        # Copy to TensorRT input buffer
        input_buffer = inputs[0]
        processed_data = torch.as_tensor(nv_processed.cuda(), device="cuda").contiguous()
        cuda.memcpy_dtod_async(input_buffer['device_mem'], processed_data.data_ptr(), processed_data.nbytes, stream)

        # Execute inference
        context.execute_v2(bindings=bindings)

        # Copy outputs
        for buffer in outputs:
            cuda.memcpy_dtoh_async(buffer['host_mem'], buffer['device_mem'], stream)

        stream.synchronize()
        inference_time = time.time() - start_time

        metrics_after = get_system_metrics()
        logger.info(f"Inference completed in {inference_time:.4f} sec")
        logger.info(f"Single query time: {(inference_time / len(batch_inputs[0])):.6f} sec")
        logger.info(f"CPU memory usage: {metrics_after['cpu_memory_mb']:.2f} MB")
        logger.info(f"GPU memory usage: {metrics_after['gpu_memory_used_mb']} MB")
        logger.info(f"CPU cores used/total: {metrics_after['cpu_cores_used']}/{metrics_after['cpu_cores_total']}")
        logger.info(f"GPU utilization: {metrics_after['gpu_utilization_percent']} %")
        logger.info(f"GPU power usage: {metrics_after['gpu_power_watts']} W")

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return None

    return [buffer['host_mem'].reshape(buffer['shape']) for buffer in outputs]

class ICFG_PEDES_Dataset(Dataset):
    """Custom Dataset for ICFG-PEDES."""
    def __init__(self, dataset_dir, csv_path):
        self.df = pd.read_csv(csv_path)
        self.dataset_dir = dataset_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.dataset_dir}/{row['image_path']}"
        if not isinstance(img_path, str) or not os.path.exists(img_path):
            logger.error(f"Invalid or missing image path: {img_path}")
            raise ValueError(f"Invalid or missing image path: {img_path}")
        img = Image.open(img_path).convert("RGB")
        return np.array(img), row['description'], row['id']

def load_icfg_pedes_data(dataset_dir, split="test", batch_size=4):
    """Load and preprocess ICFG-PEDES dataset using DataLoader and CVCUDA."""
    logger.info(f"Loading {split} data from {dataset_dir}")
    csv_path = "/media/seas_au/prsdata/xyz/cv_final/ICFG-PDES/ICFG-PEDES/captions_cleaned_local.csv"
    dataset = ICFG_PEDES_Dataset(dataset_dir, csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    cvcuda_stream = cvcuda.Stream()
    images = []
    texts = []
    labels = []

    for batch_images, batch_texts, batch_labels in tqdm(dataloader, desc=f"Loading {split} data"):
        # Convert to torch tensor and move to GPU
        batch_tensor = torch.from_numpy(np.stack(batch_images.numpy())).to("cuda").to(torch.float16)
        nv_tensor = cvcuda.as_tensor(batch_tensor, "NHWC")

        # Preprocessing with CVCUDA
        with cvcuda_stream:
            nv_resized = cvcuda.resize(nv_tensor, (len(batch_images), 224, 224, 3), cvcuda.Interp.CUBIC)
            nv_float = cvcuda.convertto(nv_resized, np.float16, scale=1/255)
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
            scale_tensor = torch.tensor([1/std[0], 1/std[1], 1/std[2]], device="cuda").reshape(1, 1, 1, 3)
            base_tensor = torch.tensor(mean, device="cuda").reshape(1, 1, 1, 3)
            nv_normalized = cvcuda.normalize(
                nv_float, base_tensor, scale_tensor, cvcuda.NormalizeFlags.SCALE_IS_STDDEV
            )

        # Convert to NumPy
        batch_processed = torch.as_tensor(nv_normalized.cuda(), device="cuda").cpu().numpy()
        images.extend(batch_processed)
        texts.extend(batch_texts)
        labels.extend(batch_labels.numpy())

        # Clear GPU memory
        del batch_images, batch_tensor, nv_tensor, nv_resized, nv_float, nv_normalized
        torch.cuda.empty_cache()
        gc.collect()

    logger.info(f"Loaded {len(images)} images, {len(texts)} texts, {len(labels)} labels")
    return np.array(images, dtype=np.float16), texts, np.array(labels)

def tokenize_texts(texts, max_length=77):
    """Tokenize text inputs using CLIP."""
    logger.info("Tokenizing texts")
    tokens = []
    for text in tqdm(texts, desc="Tokenizing texts"):
        if not isinstance(text, str):
            logger.error(f"Invalid text: {text}")
            raise ValueError(f"Invalid text: {text}")
        token = clip.tokenize([text], truncate=True).numpy().squeeze()
        tokens.append(token)
    logger.info(f"Tokenized {len(tokens)} texts")
    return np.array(tokens, dtype=np.int64)

def get_text_embeddings(texts, batch_size=32, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Compute text embeddings using PyTorch CLIP with caching."""
    cache_file = "text_embeddings.h5"
    if os.path.exists(cache_file):
        logger.info("Loading cached text embeddings")
        with h5py.File(cache_file, 'r') as f:
            text_embeddings = f['embeddings'][:]
        return text_embeddings

    logger.info(f"Computing text embeddings on {device} with batch_size={batch_size}")
    model, _ = clip.load("ViT-B/16", device=device)
    model = model.to(torch.float16)
    text_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Text embedding batches"):
        batch_texts = texts[i:i+batch_size]
        text_inputs = clip.tokenize(batch_texts, truncate=True).to(device)
        with torch.no_grad():
            batch_embeds = model.encode_text(text_inputs).cpu().numpy()
        text_embeddings.append(batch_embeds)

    text_embeddings = np.concatenate(text_embeddings, axis=0)
    logger.info(f"Generated text embeddings with shape {text_embeddings.shape}")

    with h5py.File(cache_file, 'w') as f:
        f.create_dataset('embeddings', data=text_embeddings)
    return text_embeddings

def compute_cosine_similarity(image_embeds, text_embeds, batch_size=1000):
    """Compute cosine similarity using CuPy."""
    logger.info("Computing cosine similarity on GPU with CuPy")
    image_embeds_gpu = cp.asarray(image_embeds, dtype=cp.float16)
    text_embeds_gpu = cp.asarray(text_embeds, dtype=cp.float16)

    image_norms = cp.linalg.norm(image_embeds_gpu, axis=1, keepdims=True)
    text_norms = cp.linalg.norm(text_embeds_gpu, axis=1, keepdims=True)
    image_embeds_gpu = cp.where(image_norms == 0, image_embeds_gpu, image_embeds_gpu / image_norms)
    text_embeds_gpu = cp.where(text_norms == 0, text_embeds_gpu, text_embeds_gpu / text_norms)

    similarity_matrix = np.zeros((len(image_embeds), len(text_embeds)), dtype=np.float16)
    for i in tqdm(range(0, len(image_embeds), batch_size), desc="Computing similarity"):
        batch_images = image_embeds_gpu[i:i+batch_size]
        batch_sim = cp.dot(batch_images, text_embeds_gpu.T)
        similarity_matrix[i:i+batch_size] = cp.asnumpy(batch_sim)

    logger.info(f"Cosine similarity matrix shape: {similarity_matrix.shape}")
    return similarity_matrix

def evaluate_recall_mAP(similarity_matrix, labels, k_list=[1, 5, 10]):
    """Evaluate recall and mean average precision."""
    logger.info("Evaluating recall and mAP")
    recall_at_k = {k: 0 for k in k_list}
    avg_precision_list = []

    for i in range(len(labels)):
        sorted_indices = np.argsort(-similarity_matrix[i])
        gt_label = labels[i]

        for k in k_list:
            top_k_labels = labels[sorted_indices[:k]]
            if gt_label in top_k_labels:
                recall_at_k[k] += 1

        precisions = []
        relevant_found = 0
        for rank, idx in enumerate(sorted_indices, 1):
            if labels[idx] == gt_label:
                relevant_found += 1
                precisions.append(relevant_found / rank)
                break
        avg_precision = np.mean(precisions) if precisions else 0
        avg_precision_list.append(avg_precision)

    total = len(labels)
    logger.info("\n🔹 Evaluation Metrics:")
    for k in k_list:
        logger.info(f"  - Recall@{k}: {recall_at_k[k] / total:.4f}")
    logger.info(f"  - Mean Average Precision (mAP): {np.mean(avg_precision_list):.4f}")

def benchmark_engine(engine, images, batch_size=4, num_batches=32):
    """Benchmark the engine performance."""
    logger.info(f"Starting benchmark with batch_size={batch_size}, num_batches={num_batches}")
    inputs, outputs, bindings, stream, context, cleanup = allocate_buffers(engine, batch_size)

    input_names = [inputs[i]['name'] for i in range(len(inputs))]
    if len(inputs) != 1:
        logger.error(f"Expected 1 input, got {len(inputs)}: {input_names}")
        raise ValueError(f"Expected 1 input, got {len(inputs)}: {input_names}")

    total_time = 0
    num_samples = min(len(images), batch_size * num_batches)
    batch_times = []

    for i in tqdm(range(0, num_samples, batch_size), desc="Benchmarking"):
        batch_images = images[i:i+batch_size]

        if len(batch_images) < batch_size:
            pad_size = batch_size - len(batch_images)
            batch_images = np.pad(batch_images, ((0, pad_size), (0, 0), (0, 0), (0, 0)), mode="constant")

        embeddings = run_inference(context, bindings, inputs, outputs, stream, [batch_images])
        if embeddings is None:
            logger.warning("Skipping batch due to inference failure")
            continue

        batch_time = time.time() - batch_times[-1] if batch_times else 0
        batch_times.append(time.time())
        total_time += batch_time

    if total_time > 0:
        qps = num_samples / total_time
        single_query_time = total_time / num_samples
    else:
        qps = 0
        single_query_time = 0

    logger.info(f"\n🔹 Inference Benchmark:")
    logger.info(f"  - Total Samples: {num_samples}")
    logger.info(f"  - Batch Size: {batch_size}")
    logger.info(f"  - Total Time: {total_time:.4f} sec")
    logger.info(f"  - QPS (Queries/sec): {qps:.2f}")
    logger.info(f"  - Single Query Time: {single_query_time:.6f} sec")

    cleanup()
    return embeddings

if __name__ == "__main__":
    engine_path = "/media/seas_au/prsdata/xyz/cv_final/ICFG-PDES/ICFG-PEDES/final_modelint8.engine"
    dataset_dir = "/media/seas_au/prsdata/xyz/cv_final/ICFG-PDES/ICFG-PEDES"

    logger.info("Starting inference script")
    engine = load_engine(engine_path)
    logger.info("\nEngine Info:")
    print_io_info(engine)

    images, texts, labels = load_icfg_pedes_data(dataset_dir, split="test", batch_size=4)
    tokenized_texts = tokenize_texts(texts)

    last_output = benchmark_engine(engine, images, batch_size=4, num_batches=32)

    text_embeddings = get_text_embeddings(texts, batch_size=32)

    batch_size = 4
    inputs, outputs, bindings, stream, context, cleanup = allocate_buffers(engine, batch_size)

    image_embeddings = []
    for i in tqdm(range(0, len(images), batch_size), desc="Running inference"):
        batch_images = images[i:i+batch_size]

        if len(batch_images) < batch_size:
            pad_size = batch_size - len(batch_images)
            batch_images = np.pad(batch_images, ((0, pad_size), (0, 0), (0, 0), (0, 0)), mode="constant")

        embeddings = run_inference(context, bindings, inputs, outputs, stream, [batch_images])
        if embeddings is not None:
            image_embeds = embeddings[0][:len(batch_images)]
            image_embeddings.append(image_embeds)

    cleanup()
    if image_embeddings:
        image_embeddings = np.concatenate(image_embeddings, axis=0)
        logger.info(f"Generated image embeddings with shape {image_embeddings.shape}")
    else:
        logger.error("No valid image embeddings generated")
        raise ValueError("No valid image embeddings generated")

    if image_embeddings.shape[1] != text_embeddings.shape[1]:
        logger.warning(f"Embedding dimensions mismatch. Truncating image embeddings from {image_embeddings.shape[1]} to {text_embeddings.shape[1]}")
        image_embeddings = image_embeddings[:, :text_embeddings.shape[1]]

    similarity_matrix = compute_cosine_similarity(image_embeddings, text_embeddings, batch_size=1000)
    evaluate_recall_mAP(similarity_matrix, labels)

    # Cleanup
    if gpu_handle:
        pynvml.nvmlShutdown()
    del images, tokenized_texts, text_embeddings, image_embeddings, similarity_matrix
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Inference script completed")
