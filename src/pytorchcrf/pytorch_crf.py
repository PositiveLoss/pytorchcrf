"""
@Author: Feiyang Niu
@Date:   2019-11-09 20:10:56

The implementation is based on [pytorch-crf](https://github.com/kmkurn/pytorch-crf)
with only the following difference

+ Method `_viterbi_decode` that decodes the most probable path get optimized.
  Running time gets reduced to 50% or less with batch size 15+ and sequence
  length 20+
+ The class now supports decoding top-N most probable paths through the implementation
  of the method `_viterbi_decode_nbest`
"""

import torch


class CRF(torch.nn.Module):
    """Conditional random field.
    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.
    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.
    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.
    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.
    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if isinstance(num_tags, bool) or not isinstance(num_tags, int) or num_tags <= 0:
            raise ValueError(f"invalid number of tags: {num_tags}")
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = torch.nn.Parameter(torch.empty(num_tags))
        self.end_transitions = torch.nn.Parameter(torch.empty(num_tags))
        self.transitions = torch.nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        torch.nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        torch.nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        torch.nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_tags={self.num_tags})"

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor | None = None,
        reduction: str = "sum",
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.Tensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.Tensor`): Boolean or byte mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.
        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        if reduction not in ("none", "sum", "mean", "token_mean"):
            raise ValueError(f"invalid reduction: {reduction}")
        if mask is None:
            mask = torch.ones(
                emissions.shape[:2], dtype=torch.bool, device=emissions.device
            )
        mask = mask.bool()
        self._validate(emissions, tags=tags, mask=mask)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # Remove padding before logsumexp so masked NaNs cannot poison gradients.
        emissions = emissions.masked_fill(~mask.unsqueeze(-1), 0)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == "none":
            return llh
        if reduction == "sum":
            return llh.sum()
        if reduction == "mean":
            return llh.mean()
        return llh.sum() / mask.float().sum()

    @torch.no_grad()
    def decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor | None = None,
        nbest: int | None = None,
        pad_tag: int | None = None,
    ) -> torch.Tensor:
        """Find the most likely tag sequence using Viterbi algorithm.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.Tensor`): Boolean or byte mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            nbest (`int`): Positive number of distinct paths per sequence. Must not
                exceed the number of possible paths for the shortest sequence.
            pad_tag (`int`): Tag at padded positions. Often input varies in length and
                the length will be padded to the maximum length in the batch. Tags at
                the padded positions will be assigned with a padding tag, i.e. `pad_tag`
        Returns:
            A long tensor on the emissions device, containing paths in descending
            score order (ties have no guaranteed order), of shape
            (nbest, batch_size, seq_length)
        """
        if nbest is None:
            nbest = 1
        if isinstance(nbest, bool) or not isinstance(nbest, int) or nbest < 1:
            raise ValueError("nbest must be a positive integer")
        if mask is None:
            mask = torch.ones(
                emissions.shape[:2], dtype=torch.bool, device=emissions.device
            )
        mask = mask.bool()
        self._validate(emissions, mask=mask)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # Every sequence must have enough distinct paths for the requested rank.
        shortest = int(mask.sum(dim=0).min())
        path_count = 1
        for _ in range(shortest):
            path_count *= self.num_tags
            if path_count >= nbest or self.num_tags == 1:
                break
        if nbest > path_count:
            raise ValueError(
                "nbest exceeds the number of paths for the shortest sequence"
            )

        if nbest == 1:
            return self._viterbi_decode(emissions, mask, pad_tag).unsqueeze(0)
        return self._viterbi_decode_nbest(emissions, mask, nbest, pad_tag)

    def _validate(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> None:
        if emissions.dim() != 3:
            raise ValueError(
                f"emissions must have dimension of 3, got {emissions.dim()}"
            )
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f"expected last dimension of emissions is {self.num_tags}, "
                f"got {emissions.size(2)}"
            )

        if not emissions.is_floating_point():
            raise ValueError("emissions must have a floating point dtype")
        if emissions.size(0) == 0 or emissions.size(1) == 0:
            raise ValueError(
                "emissions must have nonempty sequence and batch dimensions"
            )
        if emissions.device != self.transitions.device:
            raise ValueError("emissions and CRF parameters must be on the same device")

        if tags is not None and emissions.shape[:2] != tags.shape:
            raise ValueError(
                "the first two dimensions of emissions and tags must match, "
                f"got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}"
            )

        if tags is not None:
            if tags.dtype != torch.long:
                raise ValueError("tags must have dtype torch.long")
            if tags.device != emissions.device:
                raise ValueError("tags and emissions must be on the same device")

        if mask is not None:
            if mask.device != emissions.device:
                raise ValueError("mask and emissions must be on the same device")
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    "the first two dimensions of emissions and mask must match, "
                    f"got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}"
                )
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError("mask of the first timestep must all be on")
            time_mask = mask.transpose(0, 1) if self.batch_first else mask
            if (time_mask[1:] & ~time_mask[:-1]).any():
                raise ValueError(
                    "mask must describe contiguous sequences with right padding"
                )
        if tags is not None:
            valid_tags = tags if mask is None else tags[mask]
            if ((valid_tags < 0) | (valid_tags >= self.num_tags)).any():
                raise ValueError(
                    f"unmasked tags must be between 0 and {self.num_tags - 1}"
                )

    def _compute_score(
        self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        seq_length, batch_size = tags.shape
        tags = tags.masked_fill(~mask, 0)
        batch_index = torch.arange(batch_size, device=emissions.device)

        # Start transition score and first emission
        # shape: (batch_size,)
        # Out-of-place addition preserves promotion when emissions use a wider dtype.
        score = self.start_transitions[tags[0]] + emissions[0, batch_index, tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]].masked_fill(~mask[i], 0)

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, batch_index, tags[i]].masked_fill(~mask[i], 0)

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, batch_index]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
        self, emissions: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
        pad_tag: int | None = None,
    ) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        # return: (batch_size, seq_length)
        if pad_tag is None:
            pad_tag = 0

        device = emissions.device
        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history_idx = torch.zeros(
            (seq_length, batch_size, self.num_tags), dtype=torch.long, device=device
        )
        oor_idx = torch.zeros(
            (batch_size, self.num_tags), dtype=torch.long, device=device
        )
        oor_tag = torch.full(
            (seq_length, batch_size), pad_tag, dtype=torch.long, device=device
        )

        # - score is a tensor of size (batch_size, num_tags) where for every batch,
        #   value at column j stores the score of the best tag sequence so far that ends
        #   with tag j
        # - history_idx saves where the best tags candidate transitioned from; this is used
        #   when we trace back the best tag sequence
        # - oor_idx saves the best tags candidate transitioned from at the positions
        #   where mask is 0, i.e. out of range (oor)

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(-1), next_score, score)
            indices = torch.where(mask[i].unsqueeze(-1), indices, oor_idx)
            history_idx[i - 1] = indices

        # End transition score
        # shape: (batch_size, num_tags)
        end_score = score + self.end_transitions
        _, end_tag = end_score.max(dim=1)

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1

        # insert the best tag at each sequence end (last position with mask == 1)
        history_idx = history_idx.transpose(1, 0).contiguous()
        history_idx.scatter_(
            1,
            seq_ends.view(-1, 1, 1).expand(-1, 1, self.num_tags),
            end_tag.view(-1, 1, 1).expand(-1, 1, self.num_tags),
        )
        history_idx = history_idx.transpose(1, 0).contiguous()

        # The most probable path for each sequence
        best_tags_arr = torch.zeros(
            (seq_length, batch_size), dtype=torch.long, device=device
        )
        best_tags = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        for idx in range(seq_length - 1, -1, -1):
            best_tags = torch.gather(history_idx[idx], 1, best_tags)
            best_tags_arr[idx] = best_tags.view(batch_size)

        return torch.where(mask, best_tags_arr, oor_tag).transpose(0, 1)

    def _viterbi_decode_nbest(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
        nbest: int,
        pad_tag: int | None = None,
    ) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        # return: (nbest, batch_size, seq_length)
        if pad_tag is None:
            pad_tag = 0

        device = emissions.device
        seq_length, batch_size = mask.shape
        # Only rank zero exists at the first timestep for each ending tag.
        initial_score = self.start_transitions + emissions[0]
        score = initial_score.new_full((batch_size, self.num_tags, nbest), -torch.inf)
        score[:, :, 0] = initial_score
        identity = (
            torch.arange(self.num_tags * nbest, device=device)
            .view(1, self.num_tags, nbest)
            .expand(batch_size, -1, -1)
        )
        history = []

        for i in range(1, seq_length):
            candidates = (
                score.unsqueeze(-1)
                + self.transitions.unsqueeze(1)
                + emissions[i].unsqueeze(1).unsqueeze(2)
            )
            next_score, indices = candidates.reshape(
                batch_size, self.num_tags * nbest, self.num_tags
            ).topk(nbest, dim=1)
            active = mask[i, :, None, None]
            score = torch.where(active, next_score.transpose(1, 2), score)
            history.append(torch.where(active, indices.transpose(1, 2), identity))

        end_score = score + self.end_transitions.view(1, -1, 1)
        best = end_score.reshape(batch_size, -1).topk(nbest, dim=1).indices
        paths = torch.full(
            (seq_length, batch_size, nbest), pad_tag, dtype=torch.long, device=device
        )
        for i in range(seq_length - 1, 0, -1):
            paths[i] = torch.where(mask[i, :, None], best // nbest, pad_tag)
            best = history[i - 1].reshape(batch_size, -1).gather(1, best)
        paths[0] = best // nbest
        return paths.permute(2, 1, 0)
