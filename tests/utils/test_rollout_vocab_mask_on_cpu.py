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

"""Tests of the set of token ids that the rollout excludes from sampling, and of the entropy of a masked softmax.

`RolloutVocabMask` and `get_rollout_vocab_mask` define the excluded set `B` for the vLLM rollout, the actor and
the reference model. If `B` differs between the rollout and the actor, the actor evaluates a different
distribution than the one that generated the samples. The entropy functions must return the entropy of
`softmax(z)` when some logits `z_v` are `-inf`, which is how the actor applies `B`.
"""

import pytest
import torch

import verl.utils.torch_functional as verl_F
from verl.utils.vocab_mask import RolloutVocabMask
from verl.workers.rollout.utils import get_rollout_vocab_mask


class _Tokenizer:
    """Mock-up of a Hugging Face tokenizer. `get_rollout_vocab_mask` uses only `len(tokenizer)`.

    Args:
        size: The value of `len(tokenizer)`, i.e. the number of token ids of the tokenizer.
    """

    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        return self.size


class _Processor:
    """Mock-up of a Hugging Face multimodal processor that defines an image and a video placeholder token.

    `get_vision_placeholder_token_ids` reads the attributes `image_token_id` and `video_token_id`.
    """

    image_token_id = 7
    video_token_id = 8


def test_banned_mask_covers_tail_and_banned_ids():
    """`banned_mask(V)` is `True` exactly at the ids in `banned_token_ids` and at every id `v >= vocab_size`.

    Test vector: `vocab_size=10`, `banned_token_ids=(2, 7)`, and `V=12` logits, i.e. the model has two logits
    more than the tokenizer has tokens. The expected excluded set is `{2, 7, 10, 11}`.
    """
    mask = RolloutVocabMask(vocab_size=10, banned_token_ids=(2, 7)).banned_mask(12)

    expected = torch.zeros(12, dtype=torch.bool)
    expected[[2, 7, 10, 11]] = True
    assert torch.equal(mask, expected)


def test_banned_mask_is_none_without_excluded_ids():
    """`banned_mask(V)` is `None` if the excluded set is empty, so that the engine skips the masking.

    Test vector: `vocab_size` equals the number `V=12` of logits, and `banned_token_ids` is empty.
    """
    assert RolloutVocabMask(vocab_size=12).banned_mask(12) is None


def test_banned_mask_rejects_ids_outside_the_logits():
    """`banned_mask(V)` raises `ValueError` for a banned id `>= V`, i.e. an id for which the model has no logit.

    Test vector: the banned id 12 with `V=12` logits, whose ids are 0..11.
    """
    with pytest.raises(ValueError, match="lie outside"):
        RolloutVocabMask(vocab_size=10, banned_token_ids=(12,)).banned_mask(12)


def test_vllm_rollout_vocab_mask_matches_the_server_arguments():
    """For vLLM, the excluded set consists of the ids `>= len(tokenizer)` and the image and video placeholders.

    `vLLMHttpServer` passes `vocab_size` and `banned_token_ids` of this object to vLLM, so the equality below
    fixes the set that vLLM excludes from sampling.
    """
    vocab_mask = get_rollout_vocab_mask("vllm", _Tokenizer(10), _Processor())

    assert vocab_mask == RolloutVocabMask(vocab_size=10, banned_token_ids=(7, 8))


def test_rollout_without_vocab_mask():
    """For a rollout backend other than vLLM, `get_rollout_vocab_mask` returns `None`: no id is excluded.

    The tokenizer and the processor are the ones of the vLLM test; their placeholder ids must not be excluded.
    """
    assert get_rollout_vocab_mask("sglang", _Tokenizer(10), _Processor()) is None


@pytest.mark.parametrize(
    "entropy_fn", [verl_F.entropy_from_logits, lambda logits: verl_F.entropy_from_logits_with_chunking(logits, 2)]
)
def test_entropy_of_masked_logits_is_the_entropy_of_the_renormalized_distribution(entropy_fn):
    """With logits `-inf` at the ids in `B`, the entropy function returns the entropy of `softmax(z)`, a
    distribution that is 0 on `B`, and its gradient w.r.t. the logits is finite and 0 on `B`.

    For `p = softmax(z)` with `z_v = -inf` for `v` in `B`, `p(v) = 0` on `B` and `p` equals the softmax over the
    logits outside `B`; hence the expected entropy is computed from the logits outside `B` only. A NaN in the
    entropy or in its gradient would make the entropy term of the loss NaN.

    Args:
        entropy_fn: The function under test: `entropy_from_logits`, or `entropy_from_logits_with_chunking` with
            chunks of 2 rows, so that the 5 rows span 3 chunks.
    """
    # Test vector: 5 rows of 9 random logits, with B = {1, 8}.
    torch.manual_seed(0)
    banned = torch.zeros(9, dtype=torch.bool)
    banned[[1, 8]] = True
    logits = torch.randn(5, 9).masked_fill(banned, float("-inf")).requires_grad_()

    entropy = entropy_fn(logits)
    entropy.sum().backward()

    allowed_probs = logits.detach()[:, ~banned].softmax(dim=-1)
    expected = -(allowed_probs * allowed_probs.log()).sum(dim=-1)
    torch.testing.assert_close(entropy, expected)
    assert torch.isfinite(logits.grad).all()
    assert torch.equal(logits.grad[:, banned], torch.zeros(5, 2))
