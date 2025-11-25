# coding=utf-8
# Copyright (c) 2024 Zip Zou
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""The trainer supporting multiple metrics record."""

import contextlib
import copy
import functools
import os
import time
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from packaging import version
from torch.nn import Module
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer as Optimizer
from torch.utils.data import Dataset, IterableDataset
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import *
from transformers.trainer import Trainer, has_length
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction
from transformers.training_args import TrainingArguments
from transformers.utils import is_sagemaker_mp_enabled

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

from .mixins import MultiTaskModuleMixin
from .state import AdditionalState

DataCollator = Callable[[List[Any]], Dict[str, Any]]


def _patching_module_base(module: Module, additional_state: AdditionalState):
    if (
        isinstance(module, Module)
        and hasattr(module, "supports_report_metrics")
        and module.supports_report_metrics
        and MultiTaskModuleMixin not in module.__class__.__bases__
    ):
        module.__class__.__bases__ = module.__class__.__bases__ + (
            MultiTaskModuleMixin,
        )
        module.report_metrics = partial(module.report_metrics, additional_state)


class HfMultiTaskTrainer(Trainer):

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, Module]] = None,
        args: Optional[TrainingArguments] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset, Any]] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset], Any]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Optional[Tuple[Optimizer, LambdaLR]] = (None, None),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
        model_args: Optional[Any] = None,
    ):
        self.additional_state = AdditionalState(args)
        self.model_args = model_args
        if model is not None:
            report_patching = partial(
                _patching_module_base, additional_state=self.additional_state
            )
            model.apply(report_patching)
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    def create_optimizer(self):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            global_lr = self.args.learning_rate
            use_transv = not (self.model_args.merge_module == "no_merge")

            module_lrs = {
                "vision_backbone": (
                    self.args.vision_backbone_lr
                    if self.args.vision_backbone_lr is not None
                    else global_lr
                ),
                "projector": (
                    self.args.projector_lr
                    if self.args.projector_lr is not None
                    else global_lr
                ),
                "llm_backbone": (
                    self.args.llm_backbone_lr
                    if self.args.llm_backbone_lr is not None
                    else global_lr
                ),
                "merge_module": (
                    self.args.merge_modules_lr
                    if self.args.merge_modules_lr is not None
                    else (
                        self.args.llm_backbone_lr
                        if self.args.llm_backbone_lr is not None
                        else global_lr
                    )
                ),
            }

            optimizer_grouped_parameters = []
            assigned_params = set()

            for module_name, lr in module_lrs.items():
                if not use_transv:
                    module_params = [
                        (n, p)
                        for n, p in opt_model.named_parameters()
                        if module_name in n and p.requires_grad
                    ]

                else:
                    if module_name == "llm_backbone":
                        module_params = [
                            (n, p)
                            for n, p in opt_model.named_parameters()
                            if module_name in n
                            and ("merge" not in n and "alpha" not in n)
                            and p.requires_grad
                        ]

                    elif module_name == "merge_module":
                        module_params = [
                            (n, p)
                            for n, p in opt_model.named_parameters()
                            if ("merge" in n or "alpha" in n) and p.requires_grad
                        ]
                    else:
                        module_params = [
                            (n, p)
                            for n, p in opt_model.named_parameters()
                            if module_name in n and p.requires_grad
                        ]
                decay_group = {
                    "params": [p for n, p in module_params if n in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                    "lr": lr,
                }
                no_decay_group = {
                    "params": [
                        p for n, p in module_params if n not in decay_parameters
                    ],
                    "weight_decay": 0.0,
                    "lr": lr,
                }

                if decay_group["params"]:
                    optimizer_grouped_parameters.append(decay_group)
                if no_decay_group["params"]:
                    optimizer_grouped_parameters.append(no_decay_group)

                for n, p in module_params:
                    assigned_params.add(n)

            other_params = [
                (n, p)
                for n, p in opt_model.named_parameters()
                if n not in assigned_params and p.requires_grad
            ]
            if other_params:
                decay_group = {
                    "params": [p for n, p in other_params if n in decay_parameters],
                    "weight_decay": self.args.weight_decay,
                    "lr": global_lr,
                }
                no_decay_group = {
                    "params": [p for n, p in other_params if n not in decay_parameters],
                    "weight_decay": 0.0,
                    "lr": global_lr,
                }
                if decay_group["params"]:
                    optimizer_grouped_parameters.append(decay_group)
                if no_decay_group["params"]:
                    optimizer_grouped_parameters.append(no_decay_group)

            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                    self.args, opt_model
                )

            if "params" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("params")
            if "model" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("model")
            if "optimizer_dict" in optimizer_kwargs:
                optimizer_grouped_parameters = optimizer_kwargs.pop("optimizer_dict")

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

            if (
                "bitsandbytes" in str(optimizer_cls)
                and optimizer_kwargs.get("optim_bits", None) == 8
            ):
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32}
                        )

        if self.optimizer is not None:
            print("--- Optimizer Parameter Groups ---")
            total_params = 0
            for i, param_group in enumerate(self.optimizer.param_groups):
                num_params = sum(p.numel() for p in param_group["params"])
                total_params += num_params
                print(
                    f"Group {i} | "
                    f"Name: {param_group.get('name', 'N/A')} | "
                    f"Num Params: {num_params/1e6:.2f}M | "
                    f"LR: {param_group['lr']} | "
                    f"Weight Decay: {param_group['weight_decay']}"
                )
            print(f"Total Trainable Params in Optimizer: {total_params/1e6:.2f}M")
            print("------------------------------------")

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen

        if hasattr(self, "additional_state"):
            additional_logs = self.additional_state.pop_metrics(
                gather_func=self._nested_gather
            )
        else:
            additional_logs = {}

        epoch = logs.pop("epoch", None)
        logs.update(additional_logs)
        logs["epoch"] = epoch

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(
            self.args, self.state, self.control, logs
        )
