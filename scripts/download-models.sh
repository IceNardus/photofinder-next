#!/bin/bash
# Download models from GitHub releases
# Models URL: https://github.com/IceNardus/photofinder-next/releases/download/v0.1.0/models.tar.gz

set -e

REPO="IceNardus/photofinder-next"
TAG="${1:-v0.1.0}"
MODELS_DIR="src-tauri/resources/models"
TEMP_FILE="/tmp/models.tar.gz"

mkdir -p "$MODELS_DIR"

# Download models if not exists
if [ ! -f "$MODELS_DIR/mobileclip_s2.onnx" ]; then
    echo "Downloading models..."
    curl -L "https://github.com/${REPO}/releases/download/${TAG}/models.tar.gz" -o "$TEMP_FILE"
    tar -xzf "$TEMP_FILE" -C "$(dirname "$MODELS_DIR")"
    rm -f "$TEMP_FILE"
    echo "Models downloaded successfully"
else
    echo "Models already exist, skipping download"
fi