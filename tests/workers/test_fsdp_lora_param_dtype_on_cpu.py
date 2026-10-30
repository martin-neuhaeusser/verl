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
"""Tests of ``lora_params_to_cast`` and ``get_fsdp1_wrap_modules``.

``lora_params_to_cast`` returns the trainable parameters of a wrap unit whose dtype differs from that of its frozen
parameters, together with the latter dtype, if and only if the wrap unit violates the dtype requirement of FSDP: one
dtype for all parameters of a wrap unit (FSDP1, and FSDP2 before torch 2.8), or one dtype for the trainable parameters
of a wrap unit (FSDP2 from torch 2.8 on). ``get_fsdp1_wrap_modules`` is compared with the traversal of
``torch.distributed.fsdp.wrap``.
"""

import functools

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch.distributed.fsdp.wrap import _recursive_wrap, size_based_auto_wrap_policy

from verl.utils.fsdp_utils import get_fsdp1_wrap_modules, get_fsdp_wrap_policy
from verl.workers.engine.fsdp.utils import lora_params_to_cast


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


def _ids(params) -> set[int]:
    return {id(p) for p in params}


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
    """Under FSDP2 with torch >= 2.8, no parameter is cast, even with all parameters in the root wrap unit.

    The root wrap unit holds the frozen bf16 and the trainable fp32 parameters; its trainable parameters share one
    dtype, which is all that FSDP2 requires from torch 2.8 on.

    Args:
        monkeypatch: The pytest fixture, used to set ``torch.__version__``.
        torch_version: The torch version, including a local version label as in the CUDA wheels.
    """
    monkeypatch.setattr(torch, "__version__", torch_version)

    assert lora_params_to_cast([_peft_model()], "fsdp2") == []


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp_turbo"])
def test_one_wrap_unit_with_all_params_sharing_one_dtype_casts_lora_params(strategy):
    """Under FSDP1, and FSDPTurbo, with the root as the only wrap unit, every LoRA parameter is cast to bf16.

    The root wrap unit holds the frozen bf16 and the trainable fp32 parameters, but all of its parameters must share
    one dtype: FSDP1 holds them in one ``FlatParameter``, and for FSDPTurbo this requirement is assumed.

    Args:
        strategy: The engine strategy.
    """
    module = _peft_model()
    [(dtype, params)] = lora_params_to_cast([module], strategy)

    assert dtype == torch.bfloat16
    assert _ids(params) == _ids(_lora_params(module))


def test_fsdp1_with_lora_wrap_policy_keeps_fp32_lora_params():
    """Under FSDP1 with verl's LoRA wrap policy, no parameter is cast.

    ``get_fsdp_wrap_policy(..., is_lora=True)`` wraps every leaf module with a trainable ``weight``, i.e. the
    ``lora_A`` and ``lora_B`` linear layers, into its own wrap unit. Each of these units holds only fp32 parameters
    and the root unit only bf16 parameters, so every ``FlatParameter`` has a single dtype. The first assertion
    establishes these wrap units, the second that nothing is cast.
    """
    module = _peft_model()
    wrap_modules = get_fsdp1_wrap_modules(module, get_fsdp_wrap_policy(module, is_lora=True))

    assert _ids(p for m in wrap_modules for p in m.parameters()) == _ids(_lora_params(module))
    assert lora_params_to_cast([module, *wrap_modules], "fsdp") == []


def test_fsdp2_before_torch_2_8_casts_lora_params(monkeypatch):
    """Under FSDP2 with torch 2.7, every LoRA parameter is cast: FSDP2 then requires one dtype for all parameters.

    Args:
        monkeypatch: The pytest fixture, used to set ``torch.__version__``.
    """
    monkeypatch.setattr(torch, "__version__", "2.7.1")
    module = _peft_model()
    [(dtype, params)] = lora_params_to_cast([module], "fsdp2")

    assert dtype == torch.bfloat16
    assert _ids(params) == _ids(_lora_params(module))


def test_fsdp2_with_trainable_params_of_two_dtypes_in_one_unit_casts_the_fp32_ones(monkeypatch):
    """Under FSDP2, trainable parameters of two dtypes in one wrap unit are cast to the frozen dtype.

    Test vector: in addition to the fp32 LoRA parameters, the bf16 base weight of ``out`` is trainable, as a module
    trained full-rank next to the adapter would be, and all parameters are in the root wrap unit. The fp32 LoRA
    parameters are cast; the bf16 weight already has the frozen dtype and is not.

    Args:
        monkeypatch: The pytest fixture, used to set ``torch.__version__``.
    """
    monkeypatch.setattr(torch, "__version__", "2.13.0")
    module = _peft_model()
    module.base_model.model.out.weight.requires_grad_(True)

    [(dtype, params)] = lora_params_to_cast([module], "fsdp2")

    assert dtype == torch.bfloat16
    assert _ids(params) == _ids(_lora_params(module))


def test_fsdp2_with_trainable_params_of_two_dtypes_in_two_units_keeps_fp32_lora_params(monkeypatch):
    """Under FSDP2, trainable parameters of two dtypes in two different wrap units are not cast.

    Test vector: as in the previous test, but ``out`` is a wrap unit of its own. Then the trainable parameters of
    each wrap unit share one dtype (fp32 in the root unit, bf16 in the unit of ``out``), so no cast is needed; this
    establishes that the decision is made per wrap unit.

    Args:
        monkeypatch: The pytest fixture, used to set ``torch.__version__``.
    """
    monkeypatch.setattr(torch, "__version__", "2.13.0")
    module = _peft_model()
    out = module.base_model.model.out
    out.weight.requires_grad_(True)

    assert lora_params_to_cast([module, out], "fsdp2") == []


def test_fsdp1_casts_to_the_frozen_dtype_of_each_wrap_unit():
    """Under FSDP1, the LoRA parameters of each wrap unit are cast to the dtype of that unit's frozen parameters.

    Test vector: LoRA adapters on ``q_proj`` and ``out``, whose frozen base weights are bf16 and fp16, and each of the
    two LoRA layers is a wrap unit. The expected result has one tuple per wrap unit, with the frozen dtype and the two
    LoRA parameters of that unit; a single dtype for the model would cast one of the two units to the wrong dtype.
    """
    module = get_peft_model(_TinyModel(), LoraConfig(r=2, lora_alpha=2, target_modules=["q_proj", "out"]))
    q_proj, out = module.base_model.model.q_proj, module.base_model.model.out
    out.base_layer.to(torch.float16)

    casts = lora_params_to_cast([module, q_proj, out], "fsdp")

    assert {(dtype, frozenset(_ids(params))) for dtype, params in casts} == {
        (torch.bfloat16, frozenset(_ids(_lora_params(q_proj)))),
        (torch.float16, frozenset(_ids(_lora_params(out)))),
    }


def test_wrap_unit_without_frozen_params_raises():
    """A wrap unit that needs a cast but has no frozen parameter has no dtype to cast to, so ``ValueError`` is raised.

    Test vector: all parameters are trainable, so the root wrap unit holds bf16 and fp32 trainable parameters only.
    """
    module = _peft_model()
    for param in module.parameters():
        param.requires_grad_(True)

    with pytest.raises(ValueError, match="dtypes set\\(\\)"):
        lora_params_to_cast([module], "fsdp")


def test_wrap_unit_with_frozen_params_of_two_dtypes_raises():
    """A wrap unit that needs a cast but has frozen parameters of two dtypes has no unique dtype to cast to.

    Test vector: the frozen weight of ``out`` is fp16 and that of ``q_proj`` bf16, both in the root wrap unit.
    """
    module = _peft_model()
    module.base_model.model.out.to(torch.float16)

    with pytest.raises(ValueError, match="frozen parameters"):
        lora_params_to_cast([module], "fsdp")


def test_unknown_strategy_raises():
    """A strategy other than ``"fsdp"``, ``"fsdp2"``, and ``"fsdp_turbo"`` raises ``ValueError``."""
    with pytest.raises(ValueError, match='strategy must be .* but is "megatron"'):
        lora_params_to_cast([_peft_model()], "megatron")


class _NestedModel(torch.nn.Module):
    """Two blocks of a linear layer with 64 and one with 16 parameter elements, i.e. 80 per block and 160 in total."""

    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            torch.nn.Sequential(torch.nn.Linear(8, 8, bias=False), torch.nn.Linear(8, 2, bias=False)) for _ in range(2)
        )


def _torch_fsdp1_wrap_modules(module: torch.nn.Module, auto_wrap_policy) -> list[torch.nn.Module]:
    """Return the submodules that torch's ``_recursive_wrap`` wraps, as FSDP1 calls it for the root ``module``.

    The wrapper records each module it is called with and returns the module unchanged, so the model is not modified.
    """
    wrapped = []

    def record(submodule, **kwargs):
        wrapped.append(submodule)
        return submodule

    _recursive_wrap(
        module, auto_wrap_policy, record, ignored_modules=set(), ignored_params=set(), only_wrap_children=True
    )
    return wrapped


@pytest.mark.parametrize(
    "min_num_params, expected_wrap_modules",
    [
        (50, lambda m: [block[0] for block in m.blocks]),
        (70, lambda m: list(m.blocks)),
    ],
)
def test_get_fsdp1_wrap_modules_matches_torch_for_size_policy(min_num_params, expected_wrap_modules):
    """``get_fsdp1_wrap_modules`` returns the wrap units of torch's traversal for the size-based policy.

    Test vector: the size-based policy wraps a module whose parameter elements not yet in a nested wrap unit number at
    least ``min_num_params``. With 50, each 64-element linear layer is wrapped, and the remaining 16 elements of its
    block are not; with 70, no linear layer is wrapped, and each block with its 80 elements is. The first case
    exercises the subtraction of nested wrap units from a module's element count.

    Args:
        min_num_params: The ``min_num_params`` of the size-based policy.
        expected_wrap_modules: Maps the model to its expected wrap units, derived by hand as described above.
    """
    module = _NestedModel()
    policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)

    assert get_fsdp1_wrap_modules(module, policy) == expected_wrap_modules(module)
    assert get_fsdp1_wrap_modules(module, policy) == _torch_fsdp1_wrap_modules(module, policy)


def test_get_fsdp1_wrap_modules_matches_torch_for_lora_policy():
    """``get_fsdp1_wrap_modules`` returns the wrap units of torch's traversal for verl's LoRA policy."""
    module = _peft_model()
    policy = get_fsdp_wrap_policy(module, is_lora=True)

    assert get_fsdp1_wrap_modules(module, policy) == _torch_fsdp1_wrap_modules(module, policy)


def test_get_fsdp1_wrap_modules_without_policy_is_empty():
    """Without an ``auto_wrap_policy``, FSDP1 creates only the root wrap unit, so no submodule is returned."""
    assert get_fsdp1_wrap_modules(_NestedModel(), None) == []
