print("executing first line")
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import psutil
import logging
import os
from datetime import datetime
import matplotlib.pyplot as plt
import clip
from PIL import Image
import torchvision.transforms as transforms
import gc  # Added missing import

# Initialize logging
log_dir = "/tmp/ICFG-PDES-logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Static batch size
BATCH_SIZE = 8

def load_engine(engine_file_path):
    """Load the TensorRT engine from file with timeout."""
    logger.info(f"Loading TensorRT engine from {engine_file_path}")
    try:
        with open(engine_file_path, "rb") as f:
            engine_data = f.read()
        start_time = time.time()
        timeout = 30  # 30-second timeout
        with trt.Runtime(TRT_LOGGER) as runtime:
            while time.time() - start_time < timeout:
                try:
                    engine = runtime.deserialize_cuda_engine(engine_data)
                    if engine is None:
                        logger.error("Failed to deserialize engine")
                        raise RuntimeError("Engine deserialization failed")
                    logger.info("TensorRT engine loaded successfully")
                    logger.warning("If you see a TensorRT warning about device mismatch, the engine may not be optimized for your Orin GPU. Consider rebuilding it with trtexec --onnx=final_model.onnx --saveEngine=final_modelint8.engine --int8")
                    return engine
                except Exception as e:
                    logger.warning(f"Deserialization attempt failed: {e}, retrying...")
                    time.sleep(1)
            logger.error("Engine loading timed out after 30 seconds")
            raise RuntimeError("Engine loading timed out")
    except Exception as e:
        logger.error(f"Failed to load engine: {e}")
        raise

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
    """Retrieve CPU memory usage and core usage."""
    metrics = {}
    process = psutil.Process()
    cpu_mem = process.memory_info().rss / (1024 ** 2)
    metrics["cpu_memory_mb"] = cpu_mem
    metrics["cpu_cores_total"] = psutil.cpu_count()
    try:
        metrics["cpu_cores_used"] = len(process.cpu_affinity())
    except psutil.Error:
        metrics["cpu_cores_used"] = "Unknown"
    return metrics

def allocate_buffers(engine, batch_size=BATCH_SIZE):
    """Allocate GPU and CPU buffers for engine inputs and outputs with CUDA streams."""
    logger.info(f"Allocating buffers for batch_size={batch_size}")
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    device_mems = []

    try:
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
        if context is None:
            logger.error("Failed to create execution context")
            raise RuntimeError("Execution context creation failed")
    except Exception as e:
        logger.error(f"Buffer allocation failed: {e}")
        raise

    def cleanup():
        for dm in device_mems:
            dm.free()
        stream.synchronize()
    return inputs, outputs, bindings, stream, context, cleanup

def run_inference(context, bindings, inputs, outputs, stream, batch_inputs):
    """Run inference using TensorRT."""
    logger.info(f"Running inference for batch of size {BATCH_SIZE}")
    metrics_before = get_system_metrics()

    try:
        start_time = time.time()
        input_buffer = inputs[0]
        batch_data = batch_inputs[0].astype(np.float32)
        if batch_data.shape != input_buffer['shape']:
            logger.error(f"Input shape mismatch: expected {input_buffer['shape']}, got {batch_data.shape}")
            raise ValueError("Input shape mismatch")

        cuda.memcpy_htod_async(input_buffer['device_mem'], batch_data, stream)
        context.execute_v2(bindings=bindings)

        for buffer in outputs:
            cuda.memcpy_dtoh_async(buffer['host_mem'], buffer['device_mem'], stream)

        stream.synchronize()
        inference_time = time.time() - start_time

        metrics_after = get_system_metrics()
        logger.info(f"Inference completed in {inference_time:.4f} sec")
        logger.info(f"Single query time: {(inference_time / BATCH_SIZE):.6f} sec")
        logger.info(f"CPU memory usage: {metrics_after['cpu_memory_mb']:.2f} MB")

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return None

    return [buffer['host_mem'].reshape(buffer['shape']) for buffer in outputs]

class ICFG_PEDES_Dataset(Dataset):
    """Custom Dataset for ICFG-PEDES with PIL decoding."""
    def __init__(self, dataset_dir, csv_path):
        logger.info(f"Initializing dataset with dir={dataset_dir}, csv={csv_path}")
        self.df = pd.read_csv(csv_path)
        self.dataset_dir = dataset_dir
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                std=[0.26862954, 0.26130258, 0.27577711])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.dataset_dir}/{row['image_path']}"
        if not isinstance(img_path, str) or not os.path.exists(img_path):
            logger.error(f"Invalid or missing image path: {img_path}")
            raise ValueError(f"Invalid or missing image path: {img_path}")
        img = Image.open(img_path).convert("RGB")
        img_processed = self.preprocess(img)
        return img_processed, row['description'], row['id']

def load_icfg_pedes_data(dataset_dir, split="test", batch_size=BATCH_SIZE):
    """Load and preprocess ICFG-PEDES dataset using DataLoader."""
    logger.info(f"Loading {split} data from {dataset_dir}")
    csv_path = "/media/seas_au/prsdata/xyz/cv_final/ICFG-PDES/ICFG-PEDES/temp.csv"
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        logger.info("Did you mean to use captions_cleaned_local.csv? Check your dataset directory.")
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    dataset = ICFG_PEDES_Dataset(dataset_dir, csv_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    images = []
    texts = []
    labels = []

    for batch_images, batch_texts, batch_labels in tqdm(dataloader, desc=f"Loading {split} data"):
        # Pad or truncate to static batch size
        if len(batch_images) < batch_size:
            pad_size = batch_size - len(batch_images)
            batch_images = torch.nn.functional.pad(batch_images, (0, 0, 0, 0, 0, 0, 0, pad_size))
            batch_texts = list(batch_texts) + [''] * pad_size
            batch_labels = np.pad(batch_labels, (0, pad_size), mode="constant")
        elif len(batch_images) > batch_size:
            batch_images = batch_images[:batch_size]
            batch_texts = list(batch_texts)[:batch_size]
            batch_labels = batch_labels[:batch_size]

        # Convert to NumPy
        logger.debug(f"Batch images shape: {batch_images.shape}, dtype: {batch_images.dtype}")
        batch_processed = batch_images.cpu().numpy()
        images.extend(batch_processed)
        texts.extend(batch_texts)
        labels.extend(batch_labels)

        # Clear memory with error handling
        try:
            del batch_images
            torch.cuda.empty_cache()
            gc.collect()
        except NameError as e:
            logger.error(f"Memory cleanup failed: {e}")
            torch.cuda.empty_cache()

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

def get_text_embeddings(texts, batch_size=BATCH_SIZE, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Compute text embeddings using PyTorch CLIP."""
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
    return text_embeddings

def compute_cosine_similarity(image_embeds, text_embeds, batch_size=BATCH_SIZE):
    """Compute cosine similarity using NumPy."""
    logger.info("Computing cosine similarity with NumPy")
    image_norms = np.linalg.norm(image_embeds, axis=1, keepdims=True)
    text_norms = np.linalg.norm(text_embeds, axis=1, keepdims=True)
    image_embeds = np.divide(image_embeds, image_norms, where=image_norms!=0)
    text_embeds = np.divide(text_embeds, text_norms, where=text_norms!=0)

    similarity_matrix = np.zeros((len(image_embeds), len(text_embeds)), dtype=np.float16)
    for i in tqdm(range(0, len(image_embeds), batch_size), desc="Computing similarity"):
        batch_images = image_embeds[i:i+batch_size]
        batch_sim = np.dot(batch_images, text_embeds.T)
        similarity_matrix[i:i+batch_size] = batch_sim.astype(np.float16)

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

def benchmark_engine(engine, images, batch_size=BATCH_SIZE, num_batches=16):
    """Benchmark the engine performance with static batch size."""
    logger.info(f"Starting benchmark with batch_size={batch_size}, num_batches={num_batches}")
    inputs, outputs, bindings, stream, context, cleanup = allocate_buffers(engine, batch_size)

    input_names = [inputs[i]['name'] for i in range(len(inputs))]
    if len(inputs) != 1:
        logger.error(f"Expected 1 input, got {len(inputs)}: {input_names}")
        raise ValueError(f"Expected 1 input, got {len(inputs)}: {input_names}")

    total_time = 0
    num_samples = min(len(images), batch_size * num_batches)
    cpu_mem_history = []

    for i in tqdm(range(0, num_samples, batch_size), desc="Benchmarking"):
        batch_images = images[i:i+batch_size]
        if len(batch_images) != batch_size:
            # Skip incomplete batches to maintain static size
            continue

        metrics = get_system_metrics()
        cpu_mem_history.append(metrics["cpu_memory_mb"])

        start = time.time()
        embeddings = run_inference(context, bindings, inputs, outputs, stream, [batch_images])
        if embeddings is None:
            logger.warning("Skipping batch due to inference failure")
            continue
        total_time += (time.time() - start)

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

    # Plot metrics
    plot_metrics(cpu_mem_history)
    cleanup()
    return embeddings

def plot_metrics(cpu_mem):
    """Plot CPU memory usage."""
    iterations = range(len(cpu_mem))
    plt.figure(figsize=(10, 3))
    plt.plot(iterations, cpu_mem, label="CPU Memory (MB)")
    plt.title("CPU Memory Usage")
    plt.xlabel("Iteration")
    plt.ylabel("Memory (MB)")
    plt.legend()
    plt.tight_layout()
    plt.savefig('metrics_plot.png')
    plt.close()

if __name__ == "__main__":
    logger.info("Starting inference script")
    try:
        engine_path = "/media/seas_au/prsdata/xyz/cv_final/ICFG-PDES/ICFG-PEDES/final_modelint8.engine"
        dataset_dir = "/media/seas_au/prsdata/xyz/cv_final/ICFG-PDES/ICFG-PEDES"

        logger.info("Checking CUDA availability")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

        logger.info("Loading engine")
        engine = load_engine(engine_path)
        logger.info("\nEngine Info:")
        print_io_info(engine)

        logger.info("Loading dataset")
        images, texts, labels = load_icfg_pedes_data(dataset_dir, split="test", batch_size=BATCH_SIZE)
        logger.info("Tokenizing texts")
        tokenized_texts = tokenize_texts(texts)

        logger.info("Benchmarking engine")
        last_output = benchmark_engine(engine, images, batch_size=BATCH_SIZE, num_batches=16)

        logger.info("Computing text embeddings")
        text_embeddings = get_text_embeddings(texts, batch_size=BATCH_SIZE)

        logger.info("Allocating buffers for inference")
        inputs, outputs, bindings, stream, context, cleanup = allocate_buffers(engine, batch_size=BATCH_SIZE)

        image_embeddings = []
        cpu_mem_history = []
        logger.info("Running inference loop")
        for i in tqdm(range(0, len(images), BATCH_SIZE), desc="Running inference"):
            batch_images = images[i:i+BATCH_SIZE]
            if len(batch_images) != BATCH_SIZE:
                # Skip incomplete batches
                continue

            metrics = get_system_metrics()
            cpu_mem_history.append(metrics["cpu_memory_mb"])

            embeddings = run_inference(context, bindings, inputs, outputs, stream, [batch_images])
            if embeddings is not None:
                image_embeds = embeddings[0][:BATCH_SIZE]
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

        logger.info("Computing cosine similarity")
        similarity_matrix = compute_cosine_similarity(image_embeddings, text_embeddings, batch_size=BATCH_SIZE)
        logger.info("Evaluating metrics")
        evaluate_recall_mAP(similarity_matrix, labels)

        # Plot metrics after inference
        logger.info("Plotting metrics")
        plot_metrics(cpu_mem_history)

        # Cleanup
        del images, tokenized_texts, text_embeddings, image_embeddings, similarity_matrix
        torch.cuda.empty_cache()
        try:
            gc.collect()
        except NameError:
            logger.warning("gc module not available, skipping garbage collection")
        logger.info("Inference script completed")
    except Exception as e:
        logger.error(f"Script failed: {e}")
        torch.cuda.empty_cache()  # Ensure cleanup on error
        try:
            gc.collect()
        except NameError:
            logger.warning("gc module not available, skipping garbage collection on error")
        raise
