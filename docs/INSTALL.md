# Environments

## Conda env
```bash
git clone https://github.com/xiaomi-research/TimeViper.git
cd TimeViper

conda create -n timeviper python=3.10 -y
conda activate timeviper

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.56.2
pip install vllm==0.10.2
pip install peft==0.16.0 deepspeed==0.17.2
pip install opencv-python==4.8.0.74 opencv-contrib-python==4.8.0.74

# https://github.com/state-spaces/mamba/releases/download/v2.2.5/mamba_ssm-2.2.5+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install mamba_ssm-2.2.5+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.2/causal_conv1d-1.5.2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install causal_conv1d-1.5.2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
sed -i 's/self\.dt_proj\.bias\.copy_(inv_dt)/self.dt_proj.bias.data = inv_dt.to(self.dt_proj.bias.device, dtype=self.dt_proj.bias.dtype)/' /usr/local/lib/python3.10/dist-packages/mamba_ssm/modules/mamba_simple.py

```


