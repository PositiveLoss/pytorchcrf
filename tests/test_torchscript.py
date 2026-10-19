import copy
import io
import unittest

import torch

from pytorchcrf import CRF


class Tagger(torch.nn.Module):
    """Exercise the CRF as a child of a scripted model."""

    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(4, 3)
        self.crf = CRF(3, batch_first=True)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.crf.decode(self.projection(features), mask, nbest=3, pad_tag=-1)


class TorchScriptTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_script_and_reload_match_eager_for_dynamic_inputs(self):
        for batch_first in (False, True):
            eager = CRF(3, batch_first=batch_first).eval()
            scripted = torch.jit.script(copy.deepcopy(eager))
            buffer = io.BytesIO()
            torch.jit.save(scripted, buffer)
            buffer.seek(0)
            loaded = torch.jit.load(buffer)
            for lengths, nbest in (([1], 3), ([1, 4], 3), ([2, 3, 4], 9)):
                with self.subTest(batch_first=batch_first, lengths=lengths):
                    emissions = torch.randn(max(lengths), len(lengths), 3)
                    mask = torch.arange(max(lengths))[:, None] < torch.tensor(lengths)
                    tags = torch.randint(3, mask.shape).masked_fill(~mask, -100)
                    # Padded NaNs must remain harmless in likelihood and decoding.
                    emissions[~mask] = float("nan")
                    if batch_first:
                        emissions, tags, mask = (
                            x.transpose(0, 1) for x in (emissions, tags, mask)
                        )
                    for candidate in (scripted, loaded):
                        for reduction in ("none", "sum", "mean", "token_mean"):
                            for supplied_mask in (mask, mask.byte()):
                                torch.testing.assert_close(
                                    candidate(
                                        emissions, tags, supplied_mask, reduction
                                    ),
                                    eager(emissions, tags, supplied_mask, reduction),
                                )
                        for rank in (None, 1, nbest):
                            torch.testing.assert_close(
                                candidate.decode(emissions, mask, rank, -1),
                                eager.decode(emissions, mask, rank, -1),
                            )
                    # Optional masks and decoding arguments use their defaults.
                    clean_emissions = torch.nan_to_num(emissions)
                    clean_tags = tags.clamp_min(0)
                    torch.testing.assert_close(
                        loaded(clean_emissions, clean_tags),
                        eager(clean_emissions, clean_tags),
                    )
                    torch.testing.assert_close(
                        loaded.decode(clean_emissions), eager.decode(clean_emissions)
                    )

    def test_scripted_training_gradients_match_eager(self):
        for batch_first in (False, True):
            eager = CRF(3, batch_first=batch_first).double()
            scripted = torch.jit.script(copy.deepcopy(eager))
            emissions = torch.randn(3, 2, 3, dtype=torch.double)
            tags = torch.tensor([[0, 1], [1, 2], [-100, 0]])
            mask = tags != -100
            if batch_first:
                emissions, tags, mask = (
                    x.transpose(0, 1) for x in (emissions, tags, mask)
                )
            left = emissions.detach().requires_grad_()
            right = emissions.detach().requires_grad_()
            eager_loss = -eager(left, tags, mask, "token_mean")
            script_loss = -scripted(right, tags, mask, "token_mean")
            torch.testing.assert_close(script_loss, eager_loss)
            eager_loss.backward()
            script_loss.backward()
            torch.testing.assert_close(right.grad, left.grad)
            for actual, expected in zip(scripted.parameters(), eager.parameters()):
                torch.testing.assert_close(actual.grad, expected.grad)
            # decode's no-grad scope must not affect subsequent training.
            scripted.decode(right, mask)
            self.assertTrue(torch.is_grad_enabled())
            self.assertTrue(scripted(right, tags, mask).requires_grad)

    def test_scripted_validation(self):
        scripted = torch.jit.script(CRF(3))
        emissions = torch.randn(3, 2, 3)
        tags = torch.zeros(3, 2, dtype=torch.long)
        for mask in (
            torch.zeros(3, 2, dtype=torch.bool),
            torch.tensor([[1, 1], [0, 1], [1, 1]], dtype=torch.bool),
        ):
            with self.assertRaisesRegex(torch.jit.Error, "mask"):
                scripted(emissions, tags, mask)
        for rank in (0, -1, 28):
            with self.assertRaisesRegex(torch.jit.Error, "nbest"):
                scripted.decode(emissions, nbest=rank)
        for bad_emissions, bad_tags in (
            (emissions[:, :, :2], tags),
            (emissions, tags[:, :1]),
            (emissions, tags + 3),
        ):
            with self.assertRaises(torch.jit.Error):
                scripted(bad_emissions, bad_tags)
        with self.assertRaisesRegex(torch.jit.Error, "reduction"):
            scripted(emissions, tags, reduction="invalid")

    def test_scripted_mixed_dtype_ranking(self):
        eager = CRF(2).double()
        with torch.no_grad():
            eager.start_transitions.copy_(
                torch.tensor([1.0, 1.0 + 1e-8], dtype=torch.double)
            )
            eager.transitions.zero_()
            eager.end_transitions.zero_()
        scripted = torch.jit.script(eager)
        emissions = torch.zeros(1, 1, 2)
        tags = torch.zeros(1, 1, dtype=torch.long)
        torch.testing.assert_close(scripted(emissions, tags), eager(emissions, tags))
        torch.testing.assert_close(
            scripted.decode(emissions, nbest=2), torch.tensor([[[1]], [[0]]])
        )

    def test_scripted_parent_model(self):
        eager = Tagger().eval()
        scripted = torch.jit.script(copy.deepcopy(eager))
        features = torch.randn(2, 3, 4)
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
        expected = eager(features, mask)
        torch.testing.assert_close(scripted(features, mask), expected)
        buffer = io.BytesIO()
        torch.jit.save(scripted, buffer)
        buffer.seek(0)
        loaded = torch.jit.load(buffer)
        torch.testing.assert_close(loaded(features, mask), expected)


if __name__ == "__main__":
    unittest.main()
