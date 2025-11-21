# TimeViper: A Hybrid Mamba-Transformer Vision-Language Model for Efficient Long Video Understanding

<div align="center">

<a href="https://xuboshen.github.io/TimeViper/">
  <img src="https://img.shields.io/badge/Project%20Page-TimeViper-ff69b4.svg">
</a>
<a href='https://arxiv.org/abs/2511.16595'><img src='https://img.shields.io/badge/arXiv-2511.16595-b31b1b.svg'></a>

</div>

---

# 📰 News

- **[2025.11.21]** 🚀 Initial release of the TimeViper repository.  
  The paper is available on [arXiv](https://arxiv.org/abs/2511.16595).


# 📖 Introduction

We present **TimeViper**, a hybrid Mamba-Transformer vision-language model for efficient long video understanding. 
We introduce **TransV**, the first token-transfer module that compresses vision tokens into text tokens inside the LLM, enabling the model to process over 10,000 frames.

## ✨ Key Features

- **Hybrid MLLM Architecture**  
  Integrates native hybrid LLM for long video understanding.

- **Efficient Long Video Processing**  
  Capable of handling **over 10K frames** with significantly lower memory cost compared to standard Video-LLMs.

- **Flexible Backbones**  
  Supports MLLM construction with Transformer-based LLM backbones such as **Qwen2.5** or hybrid LLMs like **Nanov2**.

- **Advanced Techniques**  
  Includes **token dropping (TD)** and **token transfer (TransV)** for training compression.

---

# 📝 TODO List

- [ ] Release model weights  
- [ ] Add full training code  
- [ ] Add inference code  
- [ ] Support training with Qwen and Nano backbones  
- [ ] Support pdrop and TransV for both training and evaluation  

---


# 🐍 Model Zoo

Model | Backbone |  Max Frames | Checkpoint
--- | --- | --- | ---
TimeViper-9B | Nanov2-9B  | 5k | Coming Soon
TimeViper-9B-w/TransV | Nanov2-9B | 10k+ | Coming Soon


---

# 🛠️ Installation

```bash
git clone https://github.com/xiaomi-research/TimeViper.git
cd TimeViper

conda create -n timeviper python=3.10 -y
conda activate timeviper

pip install vllm==0.10.2
pip install transformers==4.56.2

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

pip install peft==0.16.0 deepspeed==0.17.2
pip install opencv-python==4.8.0.74 opencv-contrib-python==4.8.0.74
```

##  🚀 Quick Start
### Training
Coming Soon.

### Evaluation
Coming Soon.



# 📄 License
This project is released under the Apache 2.0 License.

# 📚 Citation
If you find TimeViper useful for your research and applications, please cite our paper:
```
@article{xu2025timeviper,
  title={TimeViper: A Hybrid Mamba-Transformer Model for Efficient Long Video Understanding},
  author={Xu, Boshen and Xiao, Zihan and Li, Jiaze and Ju, Jianzhong and Luo, Zhenbo and Luan, Jian and Jin, Qin},
  journal={arXiv preprint arXiv:2511.16595},
  year={2025}
}
```

# 🙏 Acknowledgement
We thank the following open-source projects for their contributions: [Cobra](https://github.com/h-zhao1997/cobra), [Vamba](https://github.com/TIGER-AI-Lab/Vamba), [transformers](https://github.com/huggingface/transformers), [vllm](https://github.com/vllm-project/vllm), [mamba](https://github.com/state-spaces/mamba), [Time-R1](https://github.com/xiaomi-research/time-r1).
