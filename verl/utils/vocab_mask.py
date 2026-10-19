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
"""Definition of the token ids that a rollout backend excludes from sampling.

Let ``V`` be the number of logits the model produces per position, and let ``B`` be a subset of
``{0, ..., V - 1}``. The rollout backend samples from ``p_B = softmax(z_B)``, where ``z_B`` equals the
logit vector ``z`` with ``z_v = -inf`` for every ``v`` in ``B``. Hence ``p_B(v) = 0`` for every ``v`` in
``B``. The actor and the reference model compute log-probabilities and entropy of ``p_B`` with the
same set ``B``, so that the distribution that generates the samples and the distribution that the
training objective evaluates are identical.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RolloutVocabMask:
    """The set ``B`` of token ids that a rollout backend excludes from sampling.

    ``B = {v : v >= vocab_size} | set(banned_token_ids)``.

    Attributes:
        vocab_size: The number of token ids, counted from 0, that are not excluded on account of
            their value. Every id ``v >= vocab_size`` is in ``B``. For vLLM, this is the number of
            tokens of the tokenizer; the model may produce more logits than that.
        banned_token_ids: The ids smaller than ``vocab_size`` that are in ``B``. For vLLM, these are
            the ids of the image and video placeholder tokens.
    """

    vocab_size: int
    banned_token_ids: tuple[int, ...] = ()

    def banned_mask(self, num_logits: int) -> torch.Tensor | None:
        """Return the indicator vector of ``B`` over the logits of a model.

        Args:
            num_logits: The number ``V`` of logits that the model produces per position, i.e. the
                number of output features of its ``lm_head``.

        Returns:
            A Boolean tensor ``m`` of shape ``(num_logits,)`` with ``m[v]`` ``True`` if and only if
            ``v`` is in ``B``. ``None`` if ``B`` contains no id in ``{0, ..., num_logits - 1}``; then
            the rollout backend samples from the softmax over all ``num_logits`` logits.

        Raises:
            ValueError: If an id in ``banned_token_ids`` does not lie in ``{0, ..., num_logits - 1}``.
        """
        out_of_range = [token_id for token_id in self.banned_token_ids if not 0 <= token_id < num_logits]
        if out_of_range:
            raise ValueError(f"banned token ids {out_of_range} lie outside the {num_logits} logits of the model")
        mask = torch.zeros(num_logits, dtype=torch.bool)
        mask[self.vocab_size :] = True
        mask[list(self.banned_token_ids)] = True
        if not mask.any():
            return None
        return mask
