# Usage Guide

This guide explains how to configure various components of the TimeViper model.

- [Usage Guide](#usage-guide)
  - [1. Changing Visual Encoders (`vision_backbone_id`)](#1-changing-visual-encoders-vision_backbone_id)
  - [2. Changing MLPs (`arch_specifier`)](#2-changing-mlps-arch_specifier)
  - [3. Changing LLM Backbones (`llm_backbone_id`, `conv_template`)](#3-changing-llm-backbones-llm_backbone_id-conv_template)
  - [4. Training-free Token Dropping within LLM Layers](#4-training-free-token-dropping-within-llm-layers)

## 1. Changing Visual Encoders (`vision_backbone_id`)

You can change the visual encoder by setting the `vision_backbone_id` argument. The supported backbones are defined in `timeviper/model/vit/registry.py`.

**Common options:**
- `siglip-vit-so400m-384px` (Default)
- `dinov2-vit-l`
- `internvideo2-1b-16-224px`

**Multi-backbone support:**
You can combine multiple backbones using `+`.
- Example: `dinov2-vit-l+siglip-vit-so400m-384px`

**Usage in training script:**
```bash
python train.py --vision_backbone_id siglip-vit-so400m-384px ...
```

## 2. Changing MLPs (`arch_specifier`)

The projector architecture is controlled by the `arch_specifier` argument.

**Options:**
- `gelu_mlp`: Standard MLP projector (Default).
- `tome_mlp-{N}`: MLP with Token Merging (ToMe), where `{N}` is the number of compressed tokens, commonly set to 16.

**Usage in training script:**
```bash
python train.py --arch_specifier gelu_mlp ...
```

## 3. Changing LLM Backbones (`llm_backbone_id`, `conv_template`)

To change the LLM backbone, you need to specify both the `llm_backbone_id` and the corresponding `conv_template`.

**Supported LLMs (`llm_backbone_id`):**
Defined in `timeviper/model/llm/llm_registry.py`., example:
- `qwen2.5-7b-instruct`
- `qwen2.5-7b-base`
- `nano-9b-v2-base`
- `nano-12b-v2-base`

**Conversation Templates (`conv_template`):**
Defined in `timeviper/data/conversation.py`.
- `default` (Qwen2)
- `qwen2`
- `nano_base`

**Usage in training script:**
```bash
python train.py --llm_backbone_id qwen2.5-7b-base --conv_template qwen2 ...
```

## 4. Training-free Token Dropping within LLM Layers

TimeViper supports training-free token dropping (Pyramid Drop) within the LLM backbone to reduce computational cost. This is configured using `--use_pdrop` and `--pdrop_type`.

**Enable Token Dropping:**
```bash
python train.py --use_pdrop --pdrop_type uni_7_0.8 ...
```

**Configuration (`pdrop_type`):**
The `pdrop_type` argument defines the dropping strategy, layer, and keep ratio.
Format: `type_layernum_ratio-type_layernum_ratio-...`

-   **Type**:
    -   `uni`: Uniform sampling (selects tokens at regular intervals).
    -   `attn`: Attention-based sampling (selects tokens with highest attention importance).
-   **Layernum**: The LLM layer index (0-based) where the dropping occurs.
-   **Ratio**: The ratio of visual tokens to *keep* (0.0 to 1.0).

**Examples:**

1.  **Single-Layer Drop:**
    Keep 80% of tokens at layer 14 using uniform sampling.
    ```bash
    --pdrop_type uni_14_0.8
    ```

2.  **Multi-Layer Drop:**
    -   Layer 14: Keep 80% (Uniform)
    -   Layer 21: Keep 60% (Attention-based)
    -   Layer 30: Keep 40% (Attention-based)
    ```bash
    --pdrop_type uni_14_0.8-uni_21_0.6-attn_30_0.4
    ```