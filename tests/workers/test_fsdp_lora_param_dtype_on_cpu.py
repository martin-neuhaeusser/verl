# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests of the decision which LoRA parameters ``FSDPEngine._build_lora_module`` casts before the FSDP wrap.

The LoRA parameters keep PEFT's fp32 dtype if and only if the strategy is ``fsdp2``, torch is 2.8 or later, and all
trainable parameters share one dtype; otherwise the trainable parameters whose dtype differs from the frozen ones are
cast to the frozen dtype (see ``_lora_params_to_cast``).
"""

import pytest
import torch
from peft import LoraConfig, get_peft_model

from verl.workers.engine.fsdp.transformer_impl import _lora_params_to_cast


class _TinyModel(torch.nn.Module):
    """A model with two linear layers in bf16, as a base model loaded with ``model_dtype=bf16``."""

    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8, bias=False)
        self.out = torch.nn.Linear(8, 4, bias=False)
        self.to(torch.bfloat16)


def _peft_model() -> torch.nn.Module:
    """Return ``_TinyModel`` with a LoRA adapter on ``q_proj``: frozen bf16 base parameters, trainable fp32 LoRA."""
    return get_peft_model(_TinyModel(), LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj"]))


def _lora_params(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for name, p in module.named_parameters() if "lora_" in name]


def test_test_vector():
    """The test vector: the base parameters are frozen bf16, the two LoRA parameters are trainable fp32."""
    module = _peft_model()
    assert {(p.requires_grad, p.dtype) for p in module.parameters()} == {
        (False, torch.bfloat16),
        (True, torch.float32),
    }
    assert len(_lora_params(module)) == 2


@pytest.mark.parametrize("torch_version", ["2.8.0", "2.13.0+cu130"])
def test_fsdp2_keeps_fp32_lora_params(monkeypatch, torch_version):
    """Under FSDP2 with torch >= 2.8, no parameter is cast: the LoRA parameters stay in fp32.

    Args:
        monkeypatch: The pytest fixture, used to set ``torch.__version__``.
        torch_version: The torch version, including a local version label as in the CUDA wheels.
    """
    monkeypatch.setattr(torch, "__version__", torch_version)
    base_dtype, params = _lora_params_to_cast(_peft_model(), "fsdp2")

    assert base_dtype == torch.bfloat16
    assert params == []


def test_fsdp1_casts_lora_params():
    """Under FSDP1, every LoRA parameter is cast, since a FlatParameter holds a single dtype."""
    module = _peft_model()
    base_dtype, params = _lora_params_to_cast(module, "fsdp")

    assert base_dtype == torch.bfloat16
    assert {id(p) for p in params} == {id(p) for p in _lora_params(module)}


def test_fsdp2_before_torch_2_8_casts_lora_params(monkeypatch):
    """Under FSDP2 with torch 2.7, every LoRA parameter is cast: FSDP2 then requires one dtype for all parameters.

    Args:
        monkeypatch: The pytest fixture, used to set ``torch.__version__``.
    """
    monkeypatch.setattr(torch, "__version__", "2.7.1")
    module = _peft_model()
    _, params = _lora_params_to_cast(module, "fsdp2")

    assert {id(p) for p in params} == {id(p) for p in _lora_params(module)}


def test_fsdp2_with_trainable_params_of_two_dtypes_casts_the_fp32_ones(monkeypatch):
    """Under FSDP2, trainable parameters of two dtypes are cast to the frozen dtype, as FSDP2 requires one dtype.

    Test vector: in addition to the fp32 LoRA parameters, the bf16 base weight of ``out`` is trainable, as a module
    trained full-rank next to the adapter would be. The fp32 LoRA parameters are cast; the bf16 weight already has the
    frozen dtype and is not.

    Args:
        monkeypatch: The pytest fixture, used to set ``torch.__version__``.
    """
    monkeypatch.setattr(torch, "__version__", "2.13.0")
    module = _peft_model()
    out_weight = module.base_model.model.out.weight
    out_weight.requires_grad_(True)

    _, params = _lora_params_to_cast(module, "fsdp2")

    assert {id(p) for p in params} == {id(p) for p in _lora_params(module)}
    assert all(p is not out_weight for p in params)


def test_without_frozen_params_nothing_is_cast():
    """Without a frozen parameter there is no base dtype to cast to: the result is ``(None, [])``."""
    module = _peft_model()
    for param in module.parameters():
        param.requires_grad_(True)

    assert _lora_params_to_cast(module, "fsdp") == (None, [])
