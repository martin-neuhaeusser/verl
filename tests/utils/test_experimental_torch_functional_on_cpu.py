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

import importlib.util
from pathlib import Path

import pytest
import torch

_MODULE_PATH = Path(__file__).resolve().parents[2] / "verl" / "utils" / "experimental" / "torch_functional.py"
_SPEC = importlib.util.spec_from_file_location("_experimental_torch_functional", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
experimental_F = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(experimental_F)


@pytest.mark.parametrize("hidden_shape", [(7, 5), (2, 7, 5)])
def test_fused_linear_for_ppo_chunked_fallback_matches_torch(monkeypatch, hidden_shape):
    monkeypatch.setattr(experimental_F, "_LIGER_FUSED_LINEAR_SCALED_CROSS_ENTROPY", None)
    monkeypatch.setattr(experimental_F, "_FLASH_ATTN_CROSS_ENTROPY_AVAILABLE", False)
    torch.manual_seed(42)

    temperature = 0.7
    vocab_size = 11
    hidden = torch.randn(hidden_shape, requires_grad=True)
    weight = torch.randn(vocab_size, hidden_shape[-1], requires_grad=True)
    labels = torch.randint(vocab_size, hidden_shape[:-1], dtype=torch.int32)
    grad_log_probs = torch.randn(hidden_shape[:-1])
    grad_entropy = torch.randn(hidden_shape[:-1])

    log_probs, entropy = experimental_F.FusedLinearForPPO()(hidden, weight, labels, temperature)
    torch.autograd.backward((log_probs, entropy), (grad_log_probs, grad_entropy))

    expected_hidden = hidden.detach().clone().requires_grad_(True)
    expected_weight = weight.detach().clone().requires_grad_(True)
    logits = ((expected_hidden @ expected_weight.t()) / temperature).float()
    expected_log_probs = logits.log_softmax(dim=-1).gather(-1, labels.long().unsqueeze(-1)).squeeze(-1)
    probs = logits.softmax(dim=-1)
    expected_entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
    torch.autograd.backward(
        (expected_log_probs, expected_entropy),
        (grad_log_probs, grad_entropy),
    )

    torch.testing.assert_close(log_probs, expected_log_probs)
    torch.testing.assert_close(entropy, expected_entropy)
    torch.testing.assert_close(hidden.grad, expected_hidden.grad)
    torch.testing.assert_close(weight.grad, expected_weight.grad)


@pytest.mark.parametrize("hidden_shape", [(7, 5), (2, 7, 5)])
def test_fused_linear_for_ppo_banned_tokens_match_masked_softmax(monkeypatch, hidden_shape):
    """With a banned-token mask, the torch backend of `FusedLinearForPPO` computes the masked softmax.

    Let `z = hidden @ weight.T / temperature`, and let `z_B` equal `z` with `z_v = -inf` for every banned id
    `v`; let `p = softmax(z_B)`. The test establishes that, for every token position `i`,
    - the log-probability equals `log p_i(y_i)`, and is `-inf` for a banned label `y_i`,
    - the entropy equals `-sum_v p_i(v) log p_i(v)`, where a summand with `p_i(v) = 0` is 0,
    - the gradients w.r.t. `hidden` and `weight` equal those of the above expressions computed with autograd,
    - the rows of the gradient w.r.t. `weight` at the banned ids are 0.
    The expected values are computed with `torch.log_softmax`, `torch.softmax`, `torch.logsumexp` and autograd.

    Args:
        monkeypatch: The pytest fixture, used to select the PyTorch implementation of the kernel.
        hidden_shape: The shape of `hidden`: `(num_tokens, hidden_size)` or `(batch_size, seq_len, hidden_size)`.
    """
    # Select the PyTorch implementation: Liger does not support the mask, and the flash-attn cross-entropy
    # is a Triton kernel that does not run on CPU.
    monkeypatch.setattr(experimental_F, "_LIGER_FUSED_LINEAR_SCALED_CROSS_ENTROPY", None)
    monkeypatch.setattr(experimental_F, "_FLASH_ATTN_CROSS_ENTROPY_AVAILABLE", False)
    torch.manual_seed(0)

    # Test vector. The banned ids are an id inside the vocabulary (3) and the two largest ids (9, 10), as in
    # the rollout mask: a placeholder token id and the ids beyond the tokenizer. The label of the first
    # position is set to the banned id 3, as for a prompt position whose next token is an image placeholder.
    # chunk_size=3 is smaller than the number of token positions, so that the result of each chunk is tested.
    temperature = 0.7
    vocab_size = 11
    banned_token_mask = torch.zeros(vocab_size, dtype=torch.bool)
    banned_token_mask[[3, 9, 10]] = True
    hidden = torch.randn(hidden_shape, requires_grad=True)
    weight = torch.randn(vocab_size, hidden_shape[-1], requires_grad=True)
    labels = torch.randint(vocab_size, hidden_shape[:-1])
    labels.view(-1)[0] = 3
    # The gradients of the loss w.r.t. the two outputs, applied in the backward pass.
    grad_log_probs = torch.randn(hidden_shape[:-1])
    grad_entropy = torch.randn(hidden_shape[:-1])

    log_probs, entropy = experimental_F.FusedLinearForPPO(chunk_size=3)(
        hidden, weight, labels, temperature, banned_token_mask=banned_token_mask
    )
    torch.autograd.backward((log_probs, entropy), (grad_log_probs, grad_entropy))

    # Expected values: the same expressions on independent copies of the leaf tensors, differentiated by
    # autograd. The entropy uses H(p) = logsumexp(z_B) - sum_v p(v) z_v, with z_v replaced by 0 at the banned
    # ids, where p(v) = 0, because 0 * (-inf) is NaN.
    expected_hidden = hidden.detach().clone().requires_grad_(True)
    expected_weight = weight.detach().clone().requires_grad_(True)
    logits = ((expected_hidden @ expected_weight.t()) / temperature).float()
    logits = logits.masked_fill(banned_token_mask, float("-inf"))
    expected_log_probs = logits.log_softmax(dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    probs = logits.softmax(dim=-1)
    finite_logits = logits.masked_fill(banned_token_mask, 0.0)
    expected_entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * finite_logits, dim=-1)
    torch.autograd.backward((expected_log_probs, expected_entropy), (grad_log_probs, grad_entropy))

    assert torch.isneginf(log_probs.view(-1)[0])
    torch.testing.assert_close(log_probs, expected_log_probs)
    torch.testing.assert_close(entropy, expected_entropy)
    torch.testing.assert_close(hidden.grad, expected_hidden.grad)
    torch.testing.assert_close(weight.grad, expected_weight.grad)
    assert torch.equal(weight.grad[banned_token_mask], torch.zeros_like(weight.grad[banned_token_mask]))


def test_fused_linear_for_ppo_liger_rejects_banned_tokens(monkeypatch):
    """The Liger backend raises `NotImplementedError` for a banned-token mask instead of ignoring the mask.

    Liger computes the softmax over all ids. Ignoring the mask would return log-probabilities of a distribution
    other than the one the rollout samples from, without any error.

    Args:
        monkeypatch: The pytest fixture, used to replace the Liger kernel.
    """

    # Replace the Liger kernel by a class whose `apply` raises AssertionError. The test then distinguishes a
    # call of the kernel (AssertionError) from the rejection of the mask before the call (NotImplementedError),
    # and does not require Liger to be installed.
    class UnexpectedLigerFusedLinearScaledCrossEntropyFunction:
        @staticmethod
        def apply(*args):
            raise AssertionError("A banned-token mask must not reach Liger")

    monkeypatch.setattr(
        experimental_F,
        "_LIGER_FUSED_LINEAR_SCALED_CROSS_ENTROPY",
        UnexpectedLigerFusedLinearScaledCrossEntropyFunction,
    )
    # Test vector: hidden states of shape (batch_size=2, seq_len=3, hidden_size=5), a vocabulary of 7 ids, and
    # a mask that bans one id. The values are arbitrary: the backend must reject every mask that is not None.
    banned_token_mask = torch.zeros(7, dtype=torch.bool)
    banned_token_mask[2] = True

    with pytest.raises(NotImplementedError, match="banned_token_mask"):
        experimental_F.FusedLinearForPPO(impl_backend="liger")(
            torch.randn(2, 3, 5), torch.randn(7, 5), torch.randint(7, (2, 3)), banned_token_mask=banned_token_mask
        )


def test_fused_linear_for_ppo_dispatches_to_liger(monkeypatch):
    calls = []

    class FakeLigerFusedLinearScaledCrossEntropyFunction:
        @staticmethod
        def apply(*args):
            calls.append(args)
            hidden_states = args[0]
            token_count = hidden_states.shape[0]
            nll = torch.arange(token_count, dtype=torch.float32)
            entropy = torch.arange(token_count, dtype=hidden_states.dtype) + 10
            return nll, entropy

    monkeypatch.setattr(
        experimental_F,
        "_LIGER_FUSED_LINEAR_SCALED_CROSS_ENTROPY",
        FakeLigerFusedLinearScaledCrossEntropyFunction,
    )
    hidden = torch.randn(2, 3, 5)
    weight = torch.randn(7, 5)
    labels = torch.randint(7, (2, 3), dtype=torch.int32)

    log_probs, entropy = experimental_F.FusedLinearForPPO(impl_backend="liger")(hidden, weight, labels, temperature=0.8)

    assert len(calls) == 1
    liger_hidden, liger_weight, liger_labels, temperature, ignore_index, m_tiles, return_entropy = calls[0]
    assert liger_hidden.shape == (6, 5)
    assert liger_weight is weight
    assert liger_labels.shape == (6,)
    assert liger_labels.dtype == torch.int64
    assert temperature == 0.8
    assert ignore_index == -100
    assert m_tiles == 1
    assert return_entropy is True
    torch.testing.assert_close(log_probs, -torch.arange(6, dtype=torch.float32).reshape(2, 3))
    torch.testing.assert_close(entropy, (torch.arange(6, dtype=hidden.dtype) + 10).reshape(2, 3))


def test_fused_linear_for_ppo_torch_backend_does_not_dispatch_to_liger(monkeypatch):
    class UnexpectedLigerFusedLinearScaledCrossEntropyFunction:
        @staticmethod
        def apply(*args):
            raise AssertionError("The torch backend must not dispatch to Liger")

    monkeypatch.setattr(
        experimental_F,
        "_LIGER_FUSED_LINEAR_SCALED_CROSS_ENTROPY",
        UnexpectedLigerFusedLinearScaledCrossEntropyFunction,
    )
    monkeypatch.setattr(experimental_F, "_FLASH_ATTN_CROSS_ENTROPY_AVAILABLE", False)

    hidden = torch.randn(2, 3, 5)
    weight = torch.randn(7, 5)
    labels = torch.randint(7, (2, 3))

    log_probs, entropy = experimental_F.FusedLinearForPPO()(hidden, weight, labels)

    logits = (hidden @ weight.t()).float()
    expected_log_probs = logits.log_softmax(dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    probs = logits.softmax(dim=-1)
    expected_entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
    torch.testing.assert_close(log_probs, expected_log_probs)
    torch.testing.assert_close(entropy, expected_entropy)


def test_fused_linear_for_ppo_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported FusedLinearForPPO backend"):
        experimental_F.FusedLinearForPPO(impl_backend="unknown")


def test_fused_linear_for_ppo_fallback_preserves_chunking(monkeypatch):
    monkeypatch.setattr(experimental_F, "_LIGER_FUSED_LINEAR_SCALED_CROSS_ENTROPY", None)
    monkeypatch.setattr(experimental_F, "_FLASH_ATTN_CROSS_ENTROPY_AVAILABLE", False)
    original_forward = experimental_F._fused_linear_for_ppo_fwd
    chunk_sizes = []

    def record_chunk_size(hidden_states, *args, **kwargs):
        chunk_sizes.append(hidden_states.shape[0])
        return original_forward(hidden_states, *args, **kwargs)

    monkeypatch.setattr(experimental_F, "_fused_linear_for_ppo_fwd", record_chunk_size)
    hidden = torch.randn(1, 7, 5)
    weight = torch.randn(11, 5)
    labels = torch.randint(11, (1, 7))

    experimental_F.FusedLinearForPPO(chunk_size=3)(hidden, weight, labels)

    assert chunk_sizes == [3, 3, 1]
