#    Copyright 2025 Renmin University of China and Xiaomi Corporation.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from __future__ import annotations

from timeviper.utils.overwatch import initialize_overwatch

from .generic_vlm import GenericTimeViperVLM

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class HybridTimeViperVLM(GenericTimeViperVLM):
    """
    A unified Vision-Language Model for Hybrid, handling both training and inference.
    supporting LLMs: Falcon-H1, zamba-2, mamba, Nanov2
    """

    supports_gradient_checkpointing = True
    _is_stateful = True

    def _prepare_cache_for_generation(
        self,
        *args,
        **kwargs,
    ):
        orig_name = self.__class__.__name__
        try:
            self.__class__.__name__ = "mamba"
            return super()._prepare_cache_for_generation(
                *args,
                **kwargs,
            )
        finally:
            self.__class__.__name__ = orig_name
