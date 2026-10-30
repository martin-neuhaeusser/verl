# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import logging
import os

import torch
from packaging import version
from torch.distributed.device_mesh import init_device_mesh

from verl.utils.device import get_device_name, is_npu_available

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def apply_npu_fsdp_patches(model_config=None):
    """Apply NPU patches for FSDP backend if NPU is available."""
    if is_npu_available:
        if model_config is not None and model_config.get("use_liger", False):
            return
        from verl.models.transformers.npu_patch import apply_npu_patches

        apply_npu_patches()


def create_device_mesh(world_size, fsdp_size):
    """
    Create a device mesh for distributed training based on the world size and FSDP size.

    Args:
        world_size (int): Total number of processes in the distributed training setup.
        fsdp_size (int): Size of the Fully Sharded Data Parallel (FSDP) group.

    Returns:
        torch.distributed.device_mesh.DeviceMesh: The initialized device mesh.
    """
    device_name = get_device_name()
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh(device_name, mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        device_mesh = init_device_mesh(
            device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["ddp", "fsdp"]
        )
    return device_mesh


def get_sharding_strategy(device_mesh, zero3_enable=True):
    """
    Determine the appropriate sharding strategy based on the number of dimensions of the device mesh.

    Args:
        device_mesh (torch.distributed.device_mesh.DeviceMesh): The device mesh used for distributed training.

    Returns:
        torch.distributed.fsdp.ShardingStrategy: The sharding strategy to be used with FSDP.

    Raises:
        NotImplementedError: If the number of dimensions of the device mesh is neither 1 nor 2.
    """

    from torch.distributed.fsdp import ShardingStrategy

    if zero3_enable:
        fsdp_strategy = ShardingStrategy.FULL_SHARD
        hsdp_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        fsdp_strategy = ShardingStrategy.SHARD_GRAD_OP
        hsdp_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2

    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
        sharding_strategy = fsdp_strategy
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
        sharding_strategy = hsdp_strategy
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


def unfuse_moe_params(weights, model_type: str | None = None):
    """Expand Transformers 5 packed MoE expert tensors to vLLM checkpoint keys.

    Transformers 5 stores Qwen-style MoE experts as packed 3D parameters:
    ``mlp.experts.gate_up_proj`` with shape
    ``[num_experts, 2 * intermediate_size, hidden_size]`` and
    ``mlp.experts.down_proj`` with shape
    ``[num_experts, hidden_size, intermediate_size]``. vLLM's Qwen MoE reload
    path still accepts the original per-expert checkpoint keys during live
    weight sync, so stream those keys without materializing a full dict.
    """
    for name, tensor in weights:
        # GPT-OSS checkpoint weights use packed 3D expert tensors directly.
        # Splitting them into per-expert 2D tensors makes vLLM's GPT-OSS loader
        # index a 2D tensor as 3D during TP slicing.
        if model_type == "gpt_oss" and ".mlp.experts." in name and tensor.dim() == 3:
            yield name, tensor
            continue

        if name.endswith(".mlp.experts.gate_up_proj") and tensor.dim() == 3:
            gate, up = tensor.chunk(2, dim=1)
            base = name.removesuffix(".gate_up_proj")
            for expert_id in range(tensor.size(0)):
                yield f"{base}.{expert_id}.gate_proj.weight", gate[expert_id].contiguous()
                yield f"{base}.{expert_id}.up_proj.weight", up[expert_id].contiguous()
            continue

        if name.endswith(".mlp.experts.down_proj") and tensor.dim() == 3:
            base = name.removesuffix(".down_proj")
            for expert_id in range(tensor.size(0)):
                yield f"{base}.{expert_id}.down_proj.weight", tensor[expert_id].contiguous()
            continue

        yield name, tensor


def lora_params_to_cast(
    wrap_units: list[torch.nn.Module], strategy: str
) -> list[tuple[torch.dtype, list[torch.nn.Parameter]]]:
    """Return the minimal set of LoRA parameters that need to be cast to the dtype of the frozen parameters
    for FSDP to correctly wrap the model while trying to preserve the precision of the trainable LoRA parameters.

    ``get_peft_model`` stores the LoRA parameters in fp32 (``autocast_adapter_dtype=True``, the default), while
    the base parameters' dtype is determined during model loading (e.g. bf16). Ideally, we'd like to maintain
    the precision of the trainable (LoRA) parameters during the FSDP wrap for more accurate gradient updates.

    However, FSDP groups trainable and frozen parameters into wrap units. Depending on the FSDP and PyTorch version,
    that wrapping may require all (trainable and frozen) parameters of a wrap unit to share the same dtype:
    - FSDP1 (``strategy="fsdp"``) and FSDP2 (``strategy="fsdp2"``) before PyTorch 2.8 require each wrap unit to
      have the same dtype across its frozen and trainable parameters. In that case, the trainable LoRA parameters must
      be cast to match the frozen parameter (i.e. base model) dtype.
    - Starting from PyTorch 2.8 and FSDP2, only the trainable parameters in each wrap unit need to share the same dtype.
      Casting to the frozen parameter dtype is no longer necessary, avoiding the precision loss that occurs when casting
      down to, e.g. bf16.

    This function determines which trainable LoRA parameters need to be cast.

    Args:
        wrap_units: The modules of which FSDP makes wrap units: the model returned by ``get_peft_model``, before the
            FSDP wrap, and any of its submodules. The wrap unit of a module ``m`` holds the parameters of ``m`` and of
            those descendants of ``m`` that are neither in ``wrap_units`` nor descendants of a module in ``wrap_units``
            below ``m``.
        strategy: ``engine_config.strategy``, i.e. ``"fsdp"``, ``"fsdp2"``, or ``"fsdp_turbo"``. For ``"fsdp_turbo"``,
            all parameters of a wrap unit must share one dtype, as for ``"fsdp"``.

    Returns:
        One tuple ``(dtype, params)`` per wrap unit whose dtypes violate the requirement of ``strategy``: ``dtype`` is
        the dtype of the frozen parameters of the wrap unit, and ``params`` are the trainable parameters of the wrap
        unit to cast to ``dtype``. The list is empty if no wrap unit violates the requirement.

    Raises:
        ValueError: If ``strategy`` is none of the above, or if a wrap unit whose dtypes violate the requirement has
            no frozen parameter or frozen parameters of more than one dtype.
    """
    match strategy:
        case "fsdp" | "fsdp_turbo":
            # FSDP1 holds the parameters of a wrap unit in one FlatParameter, which has a single dtype. FSDPTurbo is an
            # external package, and verl does not inspect which parameters of a wrap unit it requires to share one
            # dtype. If all parameters of a wrap unit share one dtype, so does every subset of them, so this
            # requirement satisfies whichever one FSDPTurbo has.
            all_params_share_dtype = True
        case "fsdp2":
            # FSDPParamGroup._init_mp_dtypes requires one dtype for all parameters of a wrap unit up to torch 2.7, and
            # for the trainable parameters of a wrap unit only from torch 2.8 on.
            all_params_share_dtype = version.parse(torch.__version__).release < (2, 8)
        case _:
            raise ValueError(f'strategy must be "fsdp", "fsdp2", or "fsdp_turbo", but is "{strategy}"')
    # The modules in ``wrap_units``, for the membership test that stops the traversal of a wrap unit at the next one.
    wrap_unit_set: set[torch.nn.Module] = set(wrap_units)
    # One tuple (dtype, params) per wrap unit that needs a cast, as returned.
    casts: list[tuple[torch.dtype, list[torch.nn.Parameter]]] = []
    for unit in wrap_units:
        # Collect the parameters of the wrap unit of ``unit`` by a depth-first traversal from ``unit`` that does not
        # enter the modules of other wrap units. Each module, and hence each parameter, is visited once over all wrap
        # units. dict.fromkeys removes parameters that several modules of the wrap unit share, and keeps the order.
        modules = [unit]
        unit_params = []
        while modules:
            submodule = modules.pop()
            unit_params.extend(submodule.parameters(recurse=False))
            modules.extend(child for child in submodule.children() if child not in wrap_unit_set)
        params = list(dict.fromkeys(unit_params))
        trainable = [p for p in params if p.requires_grad]
        constrained = params if all_params_share_dtype else trainable
        # The constrained parameters of a wrap unit must share one dtype. If they have several, the trainable
        # parameters whose dtype differs from that of the frozen parameters are cast to it; afterwards, all
        # constrained parameters have this dtype. That requires the frozen parameters to share exactly one dtype.
        if len({p.dtype for p in constrained}) > 1:
            frozen_dtypes = {p.dtype for p in params if not p.requires_grad}
            if len(frozen_dtypes) != 1:
                raise ValueError(
                    f"The wrap unit {type(unit).__name__} needs its trainable parameters cast to the dtype of its "
                    f"frozen parameters, but these have the dtypes {frozen_dtypes}"
                )
            (dtype,) = frozen_dtypes
            casts.append((dtype, [p for p in trainable if p.dtype != dtype]))
    return casts
