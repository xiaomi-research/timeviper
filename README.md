# TimeViper: A Hybrid Mamba-Transformer Vision-Language Model for Efficient Long Video Understanding

<p align="center">
        🌐 <a href="https://xuboshen.github.io/TimeViper/">Project Page</a> &nbsp&nbsp | &nbsp&nbsp 📑 <a href="https://arxiv.org/abs/2511.16595">Paper</a> &nbsp&nbsp | &nbsp&nbsp  🤗 <a href="https://huggingface.co/Boshenxx/TimeViper-9B">Model</a>
<br>

</p>


<p align="center" width="100%">
<a target="_blank"><img src="assets/timeviper.png" alt="TimeViper" style="width: 80%; min-width: 200px; display: block; margin: auto;"></a>
</p>

---

# 📰 News
- **[2025.11.25]** We provide model **w/ Nano or w/ Qwen** as backbones, and evaluation codes for **MCQ (VideoMME, LVBench, MLVU, LongVideoBench, EgoSchema, MVBench, TempCompass, CGBench), TVG (Charades, ActivityNet, TVGBench), VDC (VDC), and DVC (YouCook2)** benchmarks.  
- **[2025.11.21]** 🚀 Initial release of the TimeViper repository.  The paper is available on [arXiv](https://arxiv.org/abs/2511.16595).


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

- [x] Add inference code  
- [x] Add model code  
- [x] Add training code  
- [ ] Release model weights
- [ ] Add detailed instructions for preparing data & env & evaluation & training  
- [ ] Support training with Qwen and Nano backbones  
- [ ] Support pdrop and TransV for both training and evaluation  

---


# 🐍 Model Zoo

Model | Backbone |  Max Frames | Checkpoint
--- | --- | --- | ---
TimeViper-9B | Nanov2-9B  | 5k | [Coming Soon](https://huggingface.co/Boshenxx/TimeViper-9B)
TimeViper-9B-w/TransV | Nanov2-9B | 10k+ | Coming Soon


---

# 🛠️ Installation

Environment requirements: [INSTALL.md](./docs/INSTALL.md)

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
We thank the following open-source projects for their contributions: [Cobra](https://github.com/h-zhao1997/cobra), [Vamba](https://github.com/TIGER-AI-Lab/Vamba), [transformers](https://github.com/huggingface/transformers), [vllm](https://github.com/vllm-project/vllm), [mamba](https://github.com/state-spaces/mamba), [Time-R1](https://github.com/xiaomi-research/time-r1), [VideoChat-Flash](https://github.com/OpenGVLab/VideoChat-Flash).
