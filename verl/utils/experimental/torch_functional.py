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

from typing import Optional

import torch

try:
    from liger_kernel.ops import (
        LigerFusedLinearScaledCrossEntropyFunction as _LIGER_FUSED_LINEAR_SCALED_CROSS_ENTROPY,
    )
except ModuleNotFoundError as exc:
    if exc.name != "liger_kernel":
        raise
    _LIGER_FUSED_LINEAR_SCALED_CROSS_ENTROPY = None

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    _FLASH_ATTN_CROSS_ENTROPY_AVAILABLE = True
except ImportError:
    _FLASH_ATTN_CROSS_ENTROPY_AVAILABLE = False


def _apply_banned_token_mask(
    logits: torch.FloatTensor, banned_token_mask: Optional[torch.BoolTensor]
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    """Return the logits with ``-inf`` at the banned ids, and the logits with ``0`` at the banned ids.

    Args:
        logits: Logits ``z`` of shape ``(num_tokens, vocab_size)``. Modified in place if
            ``banned_token_mask`` is not ``None``.
        banned_token_mask: Boolean tensor of shape ``(vocab_size,)`` that is ``True`` at every banned
            id, or ``None`` if no id is banned.

    Returns:
        A tuple ``(masked_logits, finite_logits)`` of two tensors of the shape of ``logits``:
        ``masked_logits`` equals ``z`` with ``z_v = -inf`` for every banned id ``v``, and
        ``finite_logits`` equals ``z`` with ``z_v = 0`` for every banned id ``v``. Both are ``logits``
        itself if ``banned_token_mask`` is ``None``.

        ``softmax(masked_logits)`` is 0 at every banned id. ``finite_logits`` replaces
        ``masked_logits`` in every product with ``softmax(masked_logits)``: such a product is then 0
        at the banned ids, whereas ``0 * (-inf)`` is NaN.
    """
    if banned_token_mask is None:
        return logits, logits
    finite_logits = logits.masked_fill(banned_token_mask, 0.0)
    masked_logits = logits.masked_fill_(banned_token_mask, float("-inf"))
    return masked_logits, finite_logits


def _fused_linear_for_ppo_fwd(
    hidden_states: torch.FloatTensor,
    vocab_weights: torch.FloatTensor,
    input_ids: torch.LongTensor,
    temperature: float = 1.0,
    banned_token_mask: Optional[torch.BoolTensor] = None,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    """Return the log-probabilities of ``input_ids`` and the entropy of the distribution ``p`` over token ids.

    ``p = softmax(z)``, where ``z = hidden_states @ vocab_weights.T / temperature``, and ``z_v = -inf``
    for every banned id ``v``.

    Args:
        hidden_states: Hidden states of shape ``(num_tokens, hidden_size)``.
        vocab_weights: Weight matrix of the ``lm_head``, of shape ``(vocab_size, hidden_size)``.
        input_ids: The token id ``y_i`` whose log-probability is returned, per token ``i``; shape ``(num_tokens,)``.
        temperature: Positive scalar that divides the logits.
        banned_token_mask: Boolean tensor of shape ``(vocab_size,)`` that is ``True`` at every banned
            id, or ``None`` if no id is banned.

    Returns:
        A tuple ``(token_log_probs, entropy)`` of two tensors of shape ``(num_tokens,)``:
        ``token_log_probs[i] = log p_i(y_i)``, which is ``-inf`` if ``y_i`` is banned, and
        ``entropy[i] = -sum_v p_i(v) log p_i(v)``, where a summand with ``p_i(v) = 0`` is 0.
    """
    logits = (hidden_states @ vocab_weights.t()) / temperature
    orig_dtype = logits.dtype
    logits, finite_logits = _apply_banned_token_mask(logits.to(torch.float32), banned_token_mask)

    # log p(v) = z_v - logsumexp(z), hence H(p) = logsumexp(z) - sum_v p(v) z_v.
    probs = logits.softmax(dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * finite_logits, dim=-1)

    if _FLASH_ATTN_CROSS_ENTROPY_AVAILABLE:
        # The flash-attn cross-entropy returns +inf for a label whose logit is -inf.
        per_token_entropy_loss = cross_entropy_loss(logits, input_ids)[0]
        token_log_probs = -per_token_entropy_loss
    else:
        # Fallback to original PyTorch implementation
        log_probs = logits.log_softmax(dim=-1)
        token_log_probs = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)

    assert token_log_probs.dtype == torch.float32
    return token_log_probs, entropy.to(orig_dtype)


def _fused_linear_for_ppo_bwd(
    dlog_probs: Optional[torch.FloatTensor],
    dentropy: Optional[torch.FloatTensor],
    hidden_states: torch.FloatTensor,
    vocab_weights: torch.FloatTensor,
    input_ids: torch.LongTensor,
    temperature: float = 1.0,
    banned_token_mask: Optional[torch.BoolTensor] = None,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    """Return the gradients of the outputs of :func:`_fused_linear_for_ppo_fwd` w.r.t. its first two arguments.

    Args:
        dlog_probs: Gradient w.r.t. ``token_log_probs``, of shape ``(num_tokens,)``, or ``None`` if it is 0.
        dentropy: Gradient w.r.t. ``entropy``, of shape ``(num_tokens,)``, or ``None`` if it is 0.
        hidden_states, vocab_weights, input_ids, temperature, banned_token_mask: The arguments of
            :func:`_fused_linear_for_ppo_fwd`.

    Returns:
        A tuple ``(dhidden_states, dvocab_weights)`` of the shapes of ``hidden_states`` and ``vocab_weights``.
        The rows of ``dvocab_weights`` at the banned ids are 0, since ``z_v = -inf`` does not depend
        on ``vocab_weights`` for a banned id ``v``.
    """
    logits = (hidden_states @ vocab_weights.t()) / temperature
    orig_dtype = logits.dtype
    logits, finite_logits = _apply_banned_token_mask(logits.to(torch.float32), banned_token_mask)

    probs = logits.softmax(dim=-1)

    dlogits = 0

    # Gradient from log_probs
    if dlog_probs is not None:
        # d log p(y) / d z_v = [v = y] - p(v). For a banned label y, z_y = -inf is a constant, so the
        # term [v = y] is omitted.
        one_hot_input = torch.zeros_like(logits).scatter_(-1, input_ids.unsqueeze(-1), 1)
        if banned_token_mask is not None:
            one_hot_input.masked_fill_(banned_token_mask, 0.0)
        dlogits += dlog_probs.to(torch.float32).unsqueeze(-1) * (one_hot_input - probs)

    # Gradient from entropy
    if dentropy is not None:
        # d H(p) / d z_v = -p(v) (log p(v) + H(p)). log p(v) = z_v - logsumexp(z) is computed from
        # finite_logits, so that it is finite at a banned id v, where p(v) = 0 makes the product 0.
        logsumexp_logits = torch.logsumexp(logits, dim=-1, keepdim=True)
        log_probs = finite_logits - logsumexp_logits
        entropy = logsumexp_logits.squeeze(-1) - torch.sum(probs * finite_logits, dim=-1)
        dlogits += probs * (log_probs + entropy.unsqueeze(-1)) * (-dentropy.unsqueeze(-1))

    dlogits = dlogits.to(orig_dtype) / temperature

    dhidden_states = dlogits @ vocab_weights
    dvocab_weights = dlogits.t() @ hidden_states

    return dhidden_states, dvocab_weights


class FusedLinearForPPOFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.FloatTensor,
        vocab_weights: torch.FloatTensor,
        input_ids: torch.LongTensor,
        temperature: float = 1.0,
        chunk_size: int = 512,
        banned_token_mask: Optional[torch.BoolTensor] = None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        ctx.set_materialize_grads(False)

        # Cast to a 2D tensor of the shape [T, D] for ease of working
        orig_ndim = hidden_states.ndim
        assert orig_ndim in (2, 3), f"Invalid hidden_states shape, received {hidden_states.shape}"

        orig_batch_size = -1
        if orig_ndim == 3:
            assert input_ids.ndim == 2, f"input_ids shape doesn't match, {hidden_states.shape} {input_ids.shape}"
            orig_batch_size = hidden_states.shape[0]
            hidden_states_requires_grad = hidden_states.requires_grad
            hidden_states = hidden_states.flatten(0, 1)
            hidden_states.requires_grad_(hidden_states_requires_grad)
            input_ids = input_ids.flatten(0, 1)

        T = hidden_states.shape[0]

        # Allocate memory for outputs
        output_requires_grad = hidden_states.requires_grad or vocab_weights.requires_grad
        # Logits are upcasted to fp32 before computing log_probs, which are also fp32
        log_probs = torch.zeros(T, device=hidden_states.device, dtype=torch.float32, requires_grad=output_requires_grad)
        entropy = hidden_states.new_zeros(T, requires_grad=output_requires_grad)

        # Perform forward one chunk at a time
        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)

            chunk_log_probs, chunk_entropy = _fused_linear_for_ppo_fwd(
                hidden_states=hidden_states[chunk_start:chunk_end],
                vocab_weights=vocab_weights,
                input_ids=input_ids[chunk_start:chunk_end],
                temperature=temperature,
                banned_token_mask=banned_token_mask,
            )
            log_probs[chunk_start:chunk_end] = chunk_log_probs
            entropy[chunk_start:chunk_end] = chunk_entropy

        # Cast the output back to the original input dimension
        if orig_ndim == 3:
            log_probs = log_probs.view(orig_batch_size, -1)
            entropy = entropy.view(orig_batch_size, -1)

        ctx.save_for_backward(hidden_states, vocab_weights, input_ids)
        ctx.orig_batch_size = orig_batch_size
        ctx.orig_ndim = orig_ndim
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size
        ctx.banned_token_mask = banned_token_mask

        return log_probs, entropy

    @staticmethod
    def backward(ctx, dlog_probs: Optional[torch.FloatTensor], dentropy: Optional[torch.FloatTensor]):
        assert dlog_probs is not None or dentropy is not None

        hidden_states, vocab_weights, input_ids = ctx.saved_tensors
        orig_batch_size = ctx.orig_batch_size
        orig_ndim = ctx.orig_ndim
        temperature = ctx.temperature
        chunk_size = ctx.chunk_size
        banned_token_mask = ctx.banned_token_mask

        # Here orig_ndim refers to the orig_ndim of hidden_states
        if orig_ndim == 3:
            if dlog_probs is not None:
                dlog_probs = dlog_probs.flatten()
            if dentropy is not None:
                dentropy = dentropy.flatten()

        T = hidden_states.shape[0]

        # Allocate memory for outputs
        dhidden_states = None
        if hidden_states.requires_grad:
            dhidden_states = torch.zeros_like(hidden_states)
        dvocab_weights = None
        if vocab_weights.requires_grad:
            dvocab_weights = torch.zeros_like(vocab_weights)

        # Perform backward one chunk at a time
        for chunk_start in range(0, T, chunk_size):
            chunk_end = min(chunk_start + chunk_size, T)
            chunk_dlog_probs = None
            if dlog_probs is not None:
                chunk_dlog_probs = dlog_probs[chunk_start:chunk_end]
            chunk_dentropy = None
            if dentropy is not None:
                chunk_dentropy = dentropy[chunk_start:chunk_end]

            h, v = _fused_linear_for_ppo_bwd(
                dlog_probs=chunk_dlog_probs,
                dentropy=chunk_dentropy,
                hidden_states=hidden_states[chunk_start:chunk_end],
                vocab_weights=vocab_weights,
                input_ids=input_ids[chunk_start:chunk_end],
                temperature=temperature,
                banned_token_mask=banned_token_mask,
            )

            if hidden_states.requires_grad:
                dhidden_states[chunk_start:chunk_end] += h
            if vocab_weights.requires_grad:
                dvocab_weights += v

        # Cast the output back to the original input dimension
        if orig_ndim == 3 and hidden_states.requires_grad:
            hidden_size = hidden_states.shape[-1]
            dhidden_states = dhidden_states.view(orig_batch_size, -1, hidden_size)

        return (
            dhidden_states,  # hidden_states
            dvocab_weights,  # vocab_weights
            None,  # input_ids
            None,  # temperature
            None,  # chunk_size
            None,  # banned_token_mask
        )


class FusedLinearForPPO(torch.nn.Module):
    def __init__(self, chunk_size: int = 512, impl_backend: str = "torch"):
        super().__init__()

        if impl_backend not in ("torch", "liger"):
            raise ValueError(f"Unsupported FusedLinearForPPO backend: {impl_backend}. Choose 'torch' or 'liger'.")
        self.chunk_size = chunk_size
        self.impl_backend = impl_backend

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        vocab_weights: torch.FloatTensor,
        input_ids: torch.LongTensor,
        temperature: float = 1.0,
        banned_token_mask: Optional[torch.BoolTensor] = None,
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """Return the log-probabilities of ``input_ids`` and the entropy of the distribution ``p`` over token ids.

        For token position ``i``, the logit vector is ``z_i = W h_i / temperature``, where ``h_i`` is the
        hidden-state vector of position ``i`` and ``W`` is ``vocab_weights``; ``z_i[v]`` is the logit of
        token id ``v``. If ``banned_token_mask`` is given, ``z_i[v] = -inf`` for every banned id ``v``.
        Then ``p_i = softmax(z_i)``, so ``p_i(v) = 0`` for every banned id ``v``.

        Args:
            hidden_states: Hidden states ``h`` of shape ``(num_tokens, hidden_size)`` or
                ``(batch_size, seq_len, hidden_size)``.
            vocab_weights: Weight matrix ``W`` of the ``lm_head``, of shape ``(vocab_size, hidden_size)``.
            input_ids: The token id ``y_i`` whose log-probability is returned, per position ``i``;
                shape ``hidden_states.shape[:-1]``.
            temperature: Positive scalar that divides the logits.
            banned_token_mask: Boolean tensor of shape ``(vocab_size,)`` that is ``True`` at every
                banned id, or ``None`` if no id is banned. The ``liger`` backend does not support it.

        Returns:
            A tuple ``(log_probs, entropy)`` of two tensors of shape ``input_ids.shape``:
            ``log_probs[i] = log p_i(y_i)``, which is ``-inf`` if ``y_i`` is banned, and
            ``entropy[i] = -sum_v p_i(v) log p_i(v)``, where a summand with ``p_i(v) = 0`` is 0. Both
            have zero gradient w.r.t. the rows of ``vocab_weights`` at the banned ids.

        Raises:
            ValueError: If ``banned_token_mask`` does not have shape ``(vocab_size,)``.
            NotImplementedError: If ``banned_token_mask`` is given and the backend is ``liger``.
        """
        input_ids = input_ids.to(torch.int64)
        if banned_token_mask is not None and banned_token_mask.shape != (vocab_weights.shape[0],):
            raise ValueError(
                f"banned_token_mask has shape {tuple(banned_token_mask.shape)}, "
                f"expected ({vocab_weights.shape[0]},) to match vocab_weights"
            )
        if self.impl_backend == "torch" or _LIGER_FUSED_LINEAR_SCALED_CROSS_ENTROPY is None:
            return FusedLinearForPPOFunction.apply(
                hidden_states,
                vocab_weights,
                input_ids,
                temperature,
                self.chunk_size,
                banned_token_mask,
            )
        if banned_token_mask is not None:
            raise NotImplementedError(
                "The Liger fused linear cross-entropy backend does not support banned_token_mask. "
                "Use the 'torch' fused kernels backend."
            )

        if hidden_states.ndim not in (2, 3):
            raise ValueError(f"hidden_states must be 2D or 3D, got shape {tuple(hidden_states.shape)}")
        if input_ids.shape != hidden_states.shape[:-1]:
            raise ValueError(
                f"input_ids shape {tuple(input_ids.shape)} must match hidden_states shape "
                f"{tuple(hidden_states.shape[:-1])}"
            )

        output_shape = input_ids.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        input_ids = input_ids.reshape(-1)

        nll, entropy = _LIGER_FUSED_LINEAR_SCALED_CROSS_ENTROPY.apply(
            hidden_states,
            vocab_weights,
            input_ids,
            temperature,
            -100,
            1,
            True,
        )
        log_probs = -nll

        return log_probs.reshape(output_shape), entropy.reshape(output_shape)
