#!/usr/bin/env python3
"""
Train VLAD Centroids for Patch-based Image Search

Extracts SuperPoint descriptors from training images and runs k-means++
to compute K centroids for VLAD aggregation.

Usage:
    python train_vlad.py --images /path/to/training/images --output vlad_centroids.bin --k 64
"""

import argparse
import os
import sys
import struct
import random
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


SUPERPOINT_INPUT_SIZE = 256
DESCRIPTOR_DIM = 256
MAX_DESCRIPTORS = 100000  # Max descriptors to use for training


def load_superpoint_model(model_path: str):
    """Load SuperPoint ONNX model"""
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        model_path,
        sess_options,
        providers=['CPUExecutionProvider']
    )
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    return session, input_name, output_names


def preprocess_image(img: Image.Image) -> np.ndarray:
    """Preprocess image for SuperPoint: grayscale, resize, normalize to [0, 1]"""
    # Convert to grayscale
    if img.mode != 'L':
        img = img.convert('L')

    # Resize to 256x256
    img = img.resize((SUPERPOINT_INPUT_SIZE, SUPERPOINT_INPUT_SIZE), Image.LANCZOS)

    # Convert to numpy and normalize
    arr = np.array(img, dtype=np.float32) / 255.0

    # CHW format: [1, H, W]
    arr = arr[np.newaxis, :, :]

    return arr


def extract_descriptors_from_image(session, input_name, output_names, img: Image.Image, max_kpts=1024):
    """Extract SuperPoint descriptors from a PIL Image"""
    input_data = preprocess_image(img)
    input_data = input_data.astype(np.float32)

    outputs = session.run(output_names, {input_name: input_data})

    # outputs[0]: keypoints [1, N, 2]
    # outputs[1]: descriptors [1, 256, N]
    # outputs[2]: scores [1, N]

    keypoints = outputs[0][0]  # [N, 2]
    descriptors = outputs[1][0]  # [256, N]
    scores = outputs[2][0]  # [N]

    num_detected = keypoints.shape[0]

    if num_detected == 0:
        return None

    # Sort by score and take top-k
    if num_detected > max_kpts:
        top_indices = np.argsort(scores)[-max_kpts:][::-1]
    else:
        top_indices = np.arange(num_detected)

    # Transpose descriptors from [256, N] to [N, 256]
    descriptors = descriptors.T  # [N, 256]

    selected_desc = descriptors[top_indices]

    return selected_desc


def kmeans_pp_init(descriptors: np.ndarray, k: int, seed: int = 42):
    """K-means++ initialization for centroids"""
    np.random.seed(seed)
    n, dim = descriptors.shape

    centroids = np.zeros((k, dim), dtype=np.float32)

    # Choose first centroid randomly
    idx = np.random.randint(0, n)
    centroids[0] = descriptors[idx]

    # Choose remaining centroids with probability proportional to distance^2
    for c in range(1, k):
        min_dists = np.full(n, np.inf)

        for i in range(c):
            diff = descriptors - centroids[i]
            dists = np.sqrt(np.sum(diff ** 2, axis=1))
            min_dists = np.minimum(min_dists, dists)

        # Choose next centroid
        probs = min_dists ** 2
        probs /= probs.sum()

        idx = np.random.choice(n, p=probs)
        centroids[c] = descriptors[idx]

    return centroids


def kmeans(descriptors: np.ndarray, k: int, max_iterations: int = 20, seed: int = 42,
           verbose: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    K-means clustering with k-means++ initialization

    Returns:
        centroids: [K, D] cluster centroids
        assignments: [N] cluster assignment for each descriptor
    """
    n, dim = descriptors.shape

    if verbose:
        print(f"Running k-means: n={n}, k={k}, dim={dim}, max_iter={max_iterations}")

    # Initialize centroids using k-means++
    centroids = kmeans_pp_init(descriptors, k, seed)

    assignments = np.zeros(n, dtype=np.int32)

    for iteration in range(max_iterations):
        # Assign each descriptor to nearest centroid
        new_assignments = np.argmin(
            ((descriptors[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2),
            axis=1
        )

        changed = np.sum(new_assignments != assignments)
        assignments = new_assignments

        if verbose:
            print(f"  iter {iteration + 1}: {changed} reassignments")

        if changed == 0:
            if verbose:
                print(f"  Converged at iteration {iteration + 1}")
            break

        # Update centroids
        for c in range(k):
            mask = assignments == c
            if np.sum(mask) > 0:
                centroids[c] = np.mean(descriptors[mask], axis=0)

    return centroids, assignments


def save_centroids(centroids: np.ndarray, output_path: str):
    """Save centroids in binary format [K, D] f32"""
    assert centroids.dtype == np.float32

    with open(output_path, 'wb') as f:
        f.write(centroids.tobytes())

    print(f"Saved {centroids.shape[0]} centroids ({centroids.shape[1]}D) to {output_path}")


def load_centroids(input_path: str) -> np.ndarray:
    """Load centroids from binary format"""
    with open(input_path, 'rb') as f:
        data = f.read()

    num_floats = len(data) // 4
    centroids = np.frombuffer(data, dtype=np.float32)

    # Assume square matrix [K, 256]
    k = num_floats // DESCRIPTOR_DIM
    centroids = centroids.reshape(k, DESCRIPTOR_DIM)

    return centroids


def main():
    parser = argparse.ArgumentParser(description='Train VLAD centroids')
    parser.add_argument('--images', '-i', required=True,
                        help='Path to training images (directory or text file with paths)')
    parser.add_argument('--output', '-o', required=True,
                        help='Output path for centroids binary file')
    parser.add_argument('--model', '-m', default='~/Library/Caches/PhotoFinder/models-export/superpoint.onnx',
                        help='Path to SuperPoint ONNX model')
    parser.add_argument('--k', '-k', type=int, default=64,
                        help='Number of VLAD clusters (default: 64)')
    parser.add_argument('--max-descriptors', '-n', type=int, default=MAX_DESCRIPTORS,
                        help=f'Maximum descriptors to use (default: {MAX_DESCRIPTORS})')
    parser.add_argument('--max-kpts-per-patch', type=int, default=256,
                        help='Maximum keypoints per patch (default: 256)')
    parser.add_argument('--patch-size', type=int, default=512,
                        help='Patch size for splitting images (default: 512)')
    parser.add_argument('--stride', type=int, default=256,
                        help='Stride between patches (default: 256)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # Expand path
    model_path = os.path.expanduser(args.model)
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # Get list of training images
    images_path = os.path.expanduser(args.images)
    if os.path.isdir(images_path):
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(Path(images_path).glob(ext))
        image_files = [str(f) for f in image_files]
    elif os.path.isfile(images_path):
        with open(images_path) as f:
            image_files = [line.strip() for line in f if line.strip()]
    else:
        print(f"Invalid images path: {images_path}", file=sys.stderr)
        sys.exit(1)

    if len(image_files) == 0:
        print("No images found", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(image_files)} training images")

    # Load SuperPoint model
    print(f"Loading SuperPoint from {model_path}...")
    session, input_name, output_names = load_superpoint_model(model_path)
    print("Model loaded")

    # Extract descriptors from all images
    print("Extracting descriptors...")
    all_descriptors = []
    total_desc = 0

    for i, img_path in enumerate(image_files):
        if args.verbose and (i + 1) % 10 == 0:
            print(f"  Processing {i + 1}/{len(image_files)} (total desc: {total_desc})...")

        try:
            img = Image.open(img_path)
        except Exception as e:
            if args.verbose:
                print(f"  Skipping {img_path}: {e}")
            continue

        width, height = img.size

        # Calculate patch positions
        cols = 1 + (width - args.patch_size + args.stride - 1) // args.stride if width > args.patch_size else 1
        rows = 1 + (height - args.patch_size + args.stride - 1) // args.stride if height > args.patch_size else 1

        for row in range(rows):
            for col in range(cols):
                x = col * args.stride
                y = row * args.stride

                w = min(args.patch_size, width - x)
                h = min(args.patch_size, height - y)

                if w < 10 or h < 10:
                    continue

                # Crop patch
                patch = img.crop((x, y, x + w, y + h))

                # Extract descriptors from patch
                desc = extract_descriptors_from_image(
                    session, input_name, output_names, patch,
                    max_kpts=args.max_kpts_per_patch
                )
                if desc is not None and len(desc) > 0:
                    all_descriptors.append(desc)
                    total_desc += len(desc)

        # Check if we have enough descriptors
        if total_desc >= args.max_descriptors:
            if args.verbose:
                print(f"  Reached max descriptors limit at image {i + 1}")
            break

    if len(all_descriptors) == 0:
        print("No descriptors extracted", file=sys.stderr)
        sys.exit(1)

    # Concatenate all descriptors
    descriptors = np.vstack(all_descriptors)
    print(f"Extracted {descriptors.shape[0]} descriptors, shape: {descriptors.shape}")

    # Limit number of descriptors
    if descriptors.shape[0] > args.max_descriptors:
        if args.verbose:
            print(f"Sampling {args.max_descriptors} descriptors from {descriptors.shape[0]}")
        indices = random.sample(range(descriptors.shape[0]), args.max_descriptors)
        descriptors = descriptors[indices]

    # Run k-means
    print(f"Running k-means with k={args.k}...")
    centroids, assignments = kmeans(
        descriptors,
        k=args.k,
        max_iterations=20,
        seed=args.seed,
        verbose=args.verbose
    )

    # Save centroids
    save_centroids(centroids, args.output)

    # Print cluster statistics
    unique, counts = np.unique(assignments, return_counts=True)
    print(f"\nCluster distribution:")
    print(f"  Min: {counts.min()}, Max: {counts.max()}, Mean: {counts.mean():.1f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())