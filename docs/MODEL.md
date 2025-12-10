# Model preparation

- [Model preparation](#model-preparation)
  - [ViT Model Checkpoints Download Guide](#vit-model-checkpoints-download-guide)
    - [Quick Start](#quick-start)
    - [Model Checkpoints Reference Table](#model-checkpoints-reference-table)
    - [Setup Instructions](#setup-instructions)
      - [Step 1: Automated Download (Recommended)](#step-1-automated-download-recommended)
      - [Step 2: Manual Download](#step-2-manual-download)
      - [Step 3: Verify Installation](#step-3-verify-installation)
  - [LLM Model Checkpoints Download Guide](#llm-model-checkpoints-download-guide)
    - [LLM Checkpoints Reference Table](#llm-checkpoints-reference-table)
    - [Setup Instructions](#setup-instructions-1)
      - [Step 1: Create Checkpoint Directory](#step-1-create-checkpoint-directory)
      - [Step 2: Download LLM Checkpoints](#step-2-download-llm-checkpoints)
      - [Step 3: Verify Installation](#step-3-verify-installation-1)

## ViT Model Checkpoints Download Guide

This guide provides instructions for downloading and setting up SigLIP ViT model checkpoints from Hugging Face's TIMM repository.

### Quick Start

Run the automated download script to download all checkpoints at once:

```bash
chmod +x download.sh
./download.sh
```

This requires `hfd.sh` to be available in the project root directory.

### Model Checkpoints Reference Table

| Backbone ID | TIMM Model Name | Input Size | Hugging Face URL | Checkpoint File |
|---|---|---|---|---|
| siglip-vit-b16-224px | vit_base_patch16_siglip_224 | 224px | https://huggingface.co/timm/vit_base_patch16_siglip_224.v2_webli | vit_base_patch16_siglip_224.pth |
| siglip-vit-b16-256px | vit_base_patch16_siglip_256 | 256px | https://huggingface.co/timm/vit_base_patch16_siglip_256.v2_webli | vit_base_patch16_siglip_256.pth |
| siglip-vit-b16-384px | vit_base_patch16_siglip_384 | 384px | https://huggingface.co/timm/vit_base_patch16_siglip_384.v2_webli | vit_base_patch16_siglip_384.pth |
| siglip-vit-so400m | vit_so400m_patch14_siglip_224 | 224px | https://huggingface.co/timm/vit_so400m_patch14_siglip_224.v2_webli | vit_so400m_patch14_siglip_224.pth |
| siglip-vit-so400m-384px | vit_so400m_patch14_siglip_384 | 384px | https://huggingface.co/timm/vit_so400m_patch14_siglip_384.v2_webli | vit_so400m_patch14_siglip_384.pth |
| dinov2-vit-l | vit_large_patch14_reg4_dinov2.lvd142m | 224px | https://huggingface.co/timm/vit_large_patch14_dinov2.lvd142m | vit_large_patch14_reg4_dinov2.lvd142m |
| internvideo2-1b-16-224px | N/A | 224px | https://huggingface.co/OpenGVLab/InternVideo2-CLIP-1B-224p-f8 | InternVideo2-1B_f4_vision.pt |

### Setup Instructions

#### Step 1: Automated Download (Recommended)

Use the provided `download.sh` script:

```bash
./download.sh
```

#### Step 2: Manual Download

If you prefer manual download, follow these steps:

1. **Create Checkpoint Directory**

   ```bash
   mkdir -p ./ckpts/
   ```

2. **Download Checkpoints**

   For each model, visit the corresponding Hugging Face URL and download the `pytorch_model.bin` file:

   - Example: 
   - **vit_so400m_patch14_siglip_384.pth**
     - URL: https://huggingface.co/timm/vit_so400m_patch14_siglip_384.v2_webli/blob/main/pytorch_model.bin
     - Download and rename to: `vit_so400m_patch14_siglip_384.pth`

3. **Rename and Place Files**

   After downloading each `pytorch_model.bin` file, rename it according to the table above and place it in the `./ckpts/` directory.

   **Example:**
   ```bash
   mv pytorch_model.bin ./ckpts/vit_base_patch16_siglip_224.pth
   mv pytorch_model.bin ./ckpts/vit_base_patch16_siglip_256.pth
   # ...and so on for other models
   ```

   **For InternVideo2:**
   Download the model file (likely named `pytorch_model.bin` or similar) from the Hugging Face repository and rename it to `InternVideo2-1B_f4_vision.pt`.

   **For DINOv2:**
   Download the model file and rename it to `vit_large_patch14_reg4_dinov2.lvd142m`.

#### Step 3: Verify Installation

Your `./ckpts/` directory should contain all five checkpoint files:

```
./ckpts/
├── vit_base_patch16_siglip_224.pth
├── vit_base_patch16_siglip_256.pth
├── vit_base_patch16_siglip_384.pth
├── vit_so400m_patch14_siglip_224.pth
├── vit_so400m_patch14_siglip_384.pth
├── vit_large_patch14_reg4_dinov2.lvd142m
└── InternVideo2-1B_f4_vision.pt
```

The SigLIPViTBackbone class will automatically load the appropriate checkpoint during initialization.


## LLM Model Checkpoints Download Guide

This guide provides instructions for downloading and setting up LLM model checkpoints from Hugging Face.

### LLM Checkpoints Reference Table

| LLM Backbone | Hugging Face URL | Local Directory |
|---|---|---|
| Qwen2.5-7B-Base | https://huggingface.co/Qwen/Qwen2.5-7B | ./ckpts/Qwen2.5-7B-Base |
| NVIDIA-Nemotron-Nano-9B-v2-Base | https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base | ./ckpts/NVIDIA-Nemotron-Nano-9B-v2-Base |

### Setup Instructions

#### Step 1: Create Checkpoint Directory

```bash
mkdir -p ./ckpts/
```

#### Step 2: Download LLM Checkpoints

Use the `hfd.sh` tool to download the full model repositories:

**Option 1: Automated Download**

```bash
# Download Qwen2.5-7B-Base
./hfd.sh Qwen/Qwen2.5-7B -x 10 -j 10 --local-dir ./ckpts/Qwen2.5-7B-Base

# Download NVIDIA-Nemotron-Nano-9B-v2-Base
./hfd.sh nvidia/NVIDIA-Nemotron-Nano-9B-v2-Base -x 10 -j 10 --local-dir ./ckpts/NVIDIA-Nemotron-Nano-9B-v2-Base
```
#### Step 3: Verify Installation

Your `./ckpts/` directory should contain both LLM model directories:

```
./ckpts/
├── Qwen2.5-7B-Base/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.model
│   └── ...
├── NVIDIA-Nemotron-Nano-9B-v2-Base/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.model
│   └── ...
└── vit_*.pth (Vision checkpoints)
```

The model loading code will automatically load the appropriate LLM checkpoint during initialization.