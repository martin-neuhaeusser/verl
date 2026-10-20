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
"""Tests of the rollout vocabulary mask on the eager path of `FSDPEngineWithLMHead.prepare_model_outputs`.

On the eager path (`use_fused_kernels=False`), `prepare_model_outputs` receives the logits of the model, divides
them by the temperature, sets the logits of the token ids excluded by `set_rollout_vocab_mask` to `-inf`, and
computes the log-probability of each label and the entropy of the next-token distribution. The tests cover both
layouts of the logits: packed (`use_remove_padding=True`) and padded (`use_remove_padding=False`).
"""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

import verl.utils.torch_functional as verl_F
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.vocab_mask import RolloutVocabMask
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

# Test vector shared by all tests: two sequences of 3 and 2 tokens, i.e. 5 token positions, a vocabulary of 8
# ids, and a temperature different from 1, so that the division by the temperature is tested as well.
VOCAB_SIZE = 8
SEQ_LENGTHS = (3, 2)
TEMPERATURE = 0.7


def _make_engine(vocab_mask):
    """Return an `FSDPEngineWithLMHead` that provides the attributes `prepare_model_outputs` reads, and no model.

    `FSDPEngine.__init__` cannot be used: it requires an initialized `torch.distributed` process group, builds
    device meshes, and requires complete model and engine configurations. The test therefore creates the
    instance without calling `__init__` and sets the attributes that `set_rollout_vocab_mask` and the eager path
    of `prepare_model_outputs` read: no Ulysses sequence parallelism, entropy without checkpointing and without
    chunking, and a model config whose text config has `vocab_size = VOCAB_SIZE`.

    Args:
        vocab_mask: The `RolloutVocabMask` passed to `set_rollout_vocab_mask`, or `None`.
    """
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.use_ulysses_sp = False
    engine.engine_config = SimpleNamespace(entropy_checkpointing=False, entropy_from_logits_with_chunking=False)
    engine.compute_entropy_from_logits = verl_F.entropy_from_logits
    text_config = SimpleNamespace(vocab_size=VOCAB_SIZE)
    engine.model_config = SimpleNamespace(hf_config=SimpleNamespace(get_text_config=lambda: text_config))
    engine.set_rollout_vocab_mask(vocab_mask)
    return engine


def _run_prepare_model_outputs(engine, use_remove_padding, labels):
    """Return the log-probabilities and entropies that `prepare_model_outputs` computes from random logits.

    Args:
        engine: The engine returned by `_make_engine`.
        use_remove_padding: If `True`, the logits are passed in the packed layout `(1, total_nnz, VOCAB_SIZE)`;
            otherwise in the padded layout `(len(SEQ_LENGTHS), max(SEQ_LENGTHS), VOCAB_SIZE)`.
        labels: The label of each of the `total_nnz = sum(SEQ_LENGTHS)` token positions, i.e. the token id at
            the next position, of shape `(total_nnz,)`.

    Returns:
        A tuple `(log_probs, entropy, leaf)`. `leaf` is the tensor of random logits of shape
        `(total_nnz, VOCAB_SIZE)`, before the division by the temperature; it requires grad, so that the caller
        can check the gradient w.r.t. the logits. `log_probs` and `entropy` have shape `(total_nnz,)`; position
        `i` refers to row `i` of `leaf`.
    """
    total_nnz = sum(SEQ_LENGTHS)
    # `prepare_model_outputs` reads the sequence boundaries from the offsets of the jagged `input_ids`. The
    # token ids themselves are not read; they are the labels shifted by one position.
    cu_seqlens = torch.tensor([0, *torch.tensor(SEQ_LENGTHS).cumsum(0).tolist()])
    input_ids = torch.nested.nested_tensor_from_jagged(labels.roll(1), offsets=cu_seqlens)
    leaf = torch.randn(total_nnz, VOCAB_SIZE, requires_grad=True)

    # `output` replaces the output of the model: the eager path reads only its attribute `logits`.
    output = SimpleNamespace()
    output_args = {"input_ids_rmpad_rolled": labels, "temperature_is_one": False}
    if use_remove_padding:
        output.logits = leaf.unsqueeze(0)
        output_args.update(temperature_rmpad=torch.full((total_nnz,), TEMPERATURE), pad_size=0)
    else:
        # Row j of sequence b is token position cu_seqlens[b] + j; the positions after the end of a sequence
        # are padding and remain 0.
        padded = torch.zeros(len(SEQ_LENGTHS), max(SEQ_LENGTHS), VOCAB_SIZE)
        padded = padded.index_put((torch.tensor([0, 0, 0, 1, 1]), torch.tensor([0, 1, 2, 0, 1])), leaf)
        output.logits = padded
        output_args["temperature"] = torch.full((len(SEQ_LENGTHS),), TEMPERATURE)

    micro_batch = TensorDict({"input_ids": input_ids}, batch_size=[])
    tu.assign_non_tensor(
        micro_batch, use_remove_padding=use_remove_padding, pad_mode=DatasetPadMode.NO_PADDING, calculate_entropy=True
    )
    model_output = FSDPEngineWithLMHead.prepare_model_outputs(engine, output, output_args, micro_batch, None)
    return model_output["log_probs"].values(), model_output["entropy"].values(), leaf


@pytest.mark.parametrize("use_remove_padding", [True, False])
def test_eager_log_probs_and_entropy_use_the_rollout_vocabulary(use_remove_padding):
    """With a rollout vocabulary mask, the eager path computes the masked softmax.

    Let `z` be the logits divided by the temperature, `B` the set of excluded ids, and `p = softmax(z_B)`, where
    `z_B` equals `z` with `z_v = -inf` for every `v` in `B`. The test establishes that, for every position `i`,
    - the log-probability equals `log p_i(y_i)`, and is `-inf` for a label `y_i` in `B`,
    - the entropy equals the entropy of the softmax over the logits outside `B`,
    - the gradient w.r.t. the logits is finite, and 0 at the ids in `B`.
    These are the log-probabilities of the distribution from which the rollout samples.

    Args:
        use_remove_padding: The layout of the logits; see `_run_prepare_model_outputs`.
    """
    # Test vector: B = {3, 6, 7}, i.e. one id inside the vocabulary of the tokenizer (3) and the two ids beyond
    # it (vocab_size = VOCAB_SIZE - 2). The label of position 1 is the banned id 3.
    torch.manual_seed(0)
    vocab_mask = RolloutVocabMask(vocab_size=VOCAB_SIZE - 2, banned_token_ids=(3,))
    banned = vocab_mask.banned_mask(VOCAB_SIZE)
    labels = torch.tensor([1, 3, 0, 2, 5])
    engine = _make_engine(vocab_mask)

    log_probs, entropy, leaf = _run_prepare_model_outputs(engine, use_remove_padding, labels)

    # Expected values: `log_softmax` over the masked logits, and the entropy of the softmax over the logits
    # outside B.
    logits = (leaf.detach() / TEMPERATURE).masked_fill(banned, float("-inf"))
    expected_log_probs = logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    allowed_probs = logits[:, ~banned].softmax(-1)
    expected_entropy = -(allowed_probs * allowed_probs.log()).sum(-1)
    torch.testing.assert_close(log_probs, expected_log_probs)
    torch.testing.assert_close(entropy, expected_entropy)
    assert torch.isneginf(log_probs[1])

    # The log-probability of a label in B is the constant -inf and is excluded from the backward pass.
    allowed_labels = ~banned[labels]
    (log_probs[allowed_labels].sum() + entropy.sum()).backward()
    assert torch.isfinite(leaf.grad).all()
    assert torch.equal(leaf.grad[:, banned], torch.zeros_like(leaf.grad[:, banned]))


@pytest.mark.parametrize("use_remove_padding", [True, False])
def test_without_mask_the_full_vocabulary_is_used(use_remove_padding):
    """Without a rollout vocabulary mask, the eager path computes the softmax over all logits.

    The log-probability of every label equals `log_softmax(z)[y]`, where `z` is the logits divided by the
    temperature. This is the behavior for rollout backends other than vLLM, for which `get_rollout_vocab_mask`
    returns `None`.

    Args:
        use_remove_padding: The layout of the logits; see `_run_prepare_model_outputs`.
    """
    # Test vector: the labels of the masked test; the label 3 is not banned here.
    torch.manual_seed(0)
    labels = torch.tensor([1, 3, 0, 2, 5])

    log_probs, _, leaf = _run_prepare_model_outputs(_make_engine(None), use_remove_padding, labels)

    expected = (leaf.detach() / TEMPERATURE).log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(log_probs, expected)
