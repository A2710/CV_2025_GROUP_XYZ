import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
from tqdm import tqdm
import torch
from torchvision import transforms
from PIL import Image
import clip
import pandas as pd

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(engine_file_path):
    """Load the TensorRT engine from file."""
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def print_io_info(engine):
    """Print the I/O configuration of the engine."""
    print("I/O Configuration:")
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = engine.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        mode = "Input" if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "Output"
        print(f"{mode} -> Name: {name}, Shape: {shape}, Dtype: {dtype}")

def allocate_buffers(engine, batch_size=1):
    """Allocate GPU and CPU buffers for engine inputs and outputs."""
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
    return inputs, outputs, bindings, stream, context

def run_inference(context, bindings, inputs, outputs, stream, batch_inputs):
    """Run inference using the TensorRT context with CUDA."""
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

    return [buffer['host_mem'].reshape(buffer['shape']) for buffer in outputs]

def load_icfg_pedes_data(dataset_dir, split="test"):
    """Load and preprocess ICFG-PEDES dataset."""
    csv_path = "/media/seas_au/prsdata/xyz/cv_final/ICFG-PDES/ICFG-PEDES/captions_cleaned_local.csv"
    df = pd.read_csv(csv_path)
    print("Columns in CSV:", df.columns.tolist())
    
    images = []
    texts = []
    labels = []
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                             std=[0.26862954, 0.26130258, 0.27577711])
    ])

    for index, row in tqdm(df.iterrows(), desc=f"Loading {split} data", total=len(df)):
        img_path = f"{dataset_dir}/{row['image_path']}"
        if not isinstance(img_path, str):
            raise ValueError(f"Expected string for img_path, got {type(img_path)}: {img_path}")
        
        img = Image.open(img_path).convert("RGB")
        if not isinstance(img, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(img)}")
        
        img_processed = preprocess(img).numpy()
        images.append(img_processed)
        texts.append(row['description'])
        labels.append(row['id'])

    return np.array(images), texts, np.array(labels)

def tokenize_texts(texts, max_length=77):
    """Tokenize text inputs using CLIP."""
    tokens = []
    for text in tqdm(texts, desc="Tokenizing texts"):
        if not isinstance(text, str):
            raise ValueError(f"Expected string in texts, got {type(text)}: {text}")
        token = clip.tokenize([text], truncate=True).numpy().squeeze()
        tokens.append(token)
    return np.array(tokens, dtype=np.int64)

def get_text_embeddings(texts, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Compute text embeddings using PyTorch CLIP."""
    model, _ = clip.load("ViT-B/16", device=device)
    text_inputs = clip.tokenize(texts, truncate=True).to(device)
    with torch.no_grad():
        text_embeddings = model.encode_text(text_inputs).cpu().numpy()
    return text_embeddings

def compute_cosine_similarity(image_embeds, text_embeds):
    """Compute cosine similarity with handling for zero norms."""
    image_norms = np.linalg.norm(image_embeds, axis=1, keepdims=True)
    text_norms = np.linalg.norm(text_embeds, axis=1, keepdims=True)
    
    image_embeds = np.where(image_norms == 0, image_embeds, image_embeds / image_norms)
    text_embeds = np.where(text_norms == 0, text_embeds, text_embeds / text_norms)
    
    similarity = np.dot(image_embeds, text_embeds.T)
    return similarity

def evaluate_recall_mAP(similarity_matrix, labels, k_list=[1, 5, 10]):
    """Evaluate recall and mean average precision."""
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
                break  # Single relevant text per image
        avg_precision = np.mean(precisions) if precisions else 0
        avg_precision_list.append(avg_precision)

    total = len(labels)
    print("\n🔹 Evaluation Metrics:")
    for k in k_list:
        print(f"  - Recall@{k}: {recall_at_k[k] / total:.4f}")
    print(f"  - Mean Average Precision (mAP): {np.mean(avg_precision_list):.4f}")

def benchmark_engine(engine, images, batch_size=8, num_batches=100):
    """Benchmark the engine performance."""
    inputs, outputs, bindings, stream, context = allocate_buffers(engine, batch_size)
    
    input_names = [inputs[i]['name'] for i in range(len(inputs))]
    if len(inputs) != 1:
        raise ValueError(f"Expected 1 input (image), but engine has {len(inputs)} inputs: {input_names}")
    
    total_time = 0
    num_samples = min(len(images), batch_size * num_batches)

    for i in tqdm(range(0, num_samples, batch_size), desc="Benchmarking"):
        batch_images = images[i:i+batch_size]
        
        if len(batch_images) < batch_size:
            pad_size = batch_size - len(batch_images)
            batch_images = np.pad(batch_images, ((0, pad_size), (0, 0), (0, 0), (0, 0)), mode="constant")

        start = time.time()
        embeddings = run_inference(context, bindings, inputs, outputs, stream, [batch_images])
        if embeddings is None:
            print("Skipping batch due to inference failure")
            continue
        total_time += (time.time() - start)

    qps = num_samples / total_time if total_time > 0 else 0
    print(f"\n🔹 Inference Benchmark:")
    print(f"  - Total Samples: {num_samples}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Total Time: {total_time:.4f} sec")
    print(f"  - QPS (Queries/sec): {qps:.2f}")
    return embeddings

if __name__ == "__main__":
    engine_path = "/media/seas_au/prsdata/xyz/cv_final/ICFG-PDES/ICFG-PEDES/final_modelint8.engine"
    dataset_dir = "/media/seas_au/prsdata/xyz/cv_final/ICFG-PDES/ICFG-PEDES"

    engine = load_engine(engine_path)
    print("\nEngine Info:")
    print_io_info(engine)

    images, texts, labels = load_icfg_pedes_data(dataset_dir, split="test")
    tokenized_texts = tokenize_texts(texts)

    last_output = benchmark_engine(engine, images, batch_size=4, num_batches=8)

    text_embeddings = get_text_embeddings(texts)

    batch_size = 8
    inputs, outputs, bindings, stream, context = allocate_buffers(engine, batch_size)

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

    if image_embeddings:
        image_embeddings = np.concatenate(image_embeddings, axis=0)
    else:
        raise ValueError("No valid image embeddings were generated during inference")

    if image_embeddings.shape[1] != text_embeddings.shape[1]:
        print(f"Warning: Embedding dimensions mismatch. Truncating image embeddings from {image_embeddings.shape[1]} to {text_embeddings.shape[1]}")
        image_embeddings = image_embeddings[:, :text_embeddings.shape[1]]

    similarity_matrix = compute_cosine_similarity(image_embeddings, text_embeddings)
    evaluate_recall_mAP(similarity_matrix, labels)
