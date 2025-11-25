# monkeypatch

import torch

_original_torch_load = torch.load


def patched_torch_load(*args, **kwargs):
    # 默认强制 weights_only=False（如果未指定）
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)


# 打补丁
torch.load = patched_torch_load
