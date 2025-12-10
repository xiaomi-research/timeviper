#!/bin/bash

# SigLIP ViT Model Checkpoints Download Script
# This script downloads all required model checkpoints from Hugging Face TIMM repository

set -e

# Download hfd.sh tool
wget https://hf-mirror.com/hfd/hfd.sh
chmod a+x hfd.sh
export HF_ENDPOINT=https://hf-mirror.com

# Configuration
CKPTS_DIR="./ckpts"
HFD_TOOL="./hfd.sh"
NUM_WORKERS=10
NUM_JOBS=10

# Create checkpoint directory
mkdir -p "$CKPTS_DIR"

echo "Starting SigLIP ViT checkpoint downloads..."

# Array of models: (repo_name|output_filename|source_filename)
# source_filename is optional, defaults to pytorch_model.bin
declare -a MODELS=(
    "timm/vit_base_patch16_siglip_224.v2_webli|vit_base_patch16_siglip_224.pth"
    "timm/vit_base_patch16_siglip_256.v2_webli|vit_base_patch16_siglip_256.pth"
    "timm/vit_base_patch16_siglip_384.v2_webli|vit_base_patch16_siglip_384.pth"
    "timm/vit_so400m_patch14_siglip_224.v2_webli|vit_so400m_patch14_siglip_224.pth"
    "timm/vit_so400m_patch14_siglip_384.v2_webli|vit_so400m_patch14_siglip_384.pth"
    "timm/vit_large_patch14_dinov2.lvd142m|vit_large_patch14_reg4_dinov2.lvd142m"
    "OpenGVLab/InternVideo2-CLIP-1B-224p-f8|InternVideo2-1B_f4_vision.pt|1B_clip.pth"
)

# Download and rename each model
for model in "${MODELS[@]}"; do
    IFS='|' read -r repo_name output_filename source_filename <<< "$model"
    
    # Default source filename if not provided
    if [ -z "$source_filename" ]; then
        source_filename="pytorch_model.bin"
    fi

    temp_dir="$CKPTS_DIR/${repo_name##*/}"
    
    echo "Downloading: $repo_name -> $output_filename (source: $source_filename)"
    
    # Download using hfd.sh
    $HFD_TOOL "$repo_name" -x $NUM_WORKERS -j $NUM_JOBS --local-dir "$temp_dir"
    
    # Rename source file to the target filename
    if [ -f "$temp_dir/$source_filename" ]; then
        mv "$temp_dir/$source_filename" "$CKPTS_DIR/$output_filename"
        echo "✓ Successfully downloaded and renamed: $output_filename"
    else
        echo "✗ Error: $source_filename not found for $repo_name"
        exit 1
    fi
    
    # Clean up temporary directory if empty
    rmdir "$temp_dir" 2>/dev/null || true
done

echo ""
echo "All checkpoints downloaded successfully!"
echo "Checkpoint directory contents:"
ls -lh "$CKPTS_DIR"
