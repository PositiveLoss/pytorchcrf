import copy
import io
import itertools
import unittest

import torch

from pytorchcrf import CRF


def enumerate_paths(model, emissions):
    paths = list(itertools.product(range(model.num_tags), repeat=len(emissions)))
    scores = []
    for path in paths:
        score = model.start_transitions[path[0]] + emissions[0, path[0]]
        for i in range(1, len(path)):
            score = (
                score + model.transitions[path[i - 1], path[i]] + emissions[i, path[i]]
            )
        scores.append(score + model.end_transitions[path[-1]])
    return paths, torch.stack(scores)


class CRFTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def test_likelihood_and_nbest_against_enumeration(self):
        for batch_first in (False, True):
            for lengths, nbest in (
                ([1, 1], 3),
                ([1, 3], 3),
                ([2, 3], 7),
                ([2, 3], 9),
                ([3, 3], 27),
                ([3, 3], 1),
            ):
                with self.subTest(
                    batch_first=batch_first, lengths=lengths, nbest=nbest
                ):
                    model = CRF(3, batch_first=batch_first).double()
                    emissions = torch.randn(max(lengths), 2, 3, dtype=torch.double)
                    tags = torch.randint(3, (max(lengths), 2))
                    mask = torch.arange(max(lengths))[:, None] < torch.tensor(lengths)
                    inputs = (emissions, tags, mask)
                    if batch_first:
                        inputs = tuple(x.transpose(0, 1) for x in inputs)
                    actual = model(inputs[0], inputs[1], inputs[2], reduction="none")
                    decoded = model.decode(
                        inputs[0], inputs[2], nbest=nbest, pad_tag=-1
                    )
                    self.assertEqual(decoded.shape, (nbest, 2, max(lengths)))
                    self.assertEqual(decoded.dtype, torch.long)
                    self.assertFalse(decoded.requires_grad)
                    for b, length in enumerate(lengths):
                        paths, scores = enumerate_paths(model, emissions[:length, b])
                        gold = paths.index(tuple(tags[:length, b].tolist()))
                        torch.testing.assert_close(
                            actual[b], scores[gold] - scores.logsumexp(0)
                        )
                        expected = torch.tensor(
                            [paths[i] for i in scores.topk(nbest).indices.tolist()]
                        )
                        torch.testing.assert_close(decoded[:, b, :length], expected)
                        self.assertTrue((decoded[:, b, length:] == -1).all())
                    for reduction, expected in (
                        ("sum", actual.sum()),
                        ("mean", actual.mean()),
                        ("token_mean", actual.sum() / sum(lengths)),
                    ):
                        torch.testing.assert_close(
                            model(*inputs, reduction=reduction), expected
                        )

    def test_padding_and_byte_masks(self):
        model = CRF(2)
        emissions = torch.randn(3, 2, 2, requires_grad=True)
        mask = torch.tensor([[1, 1], [0, 1], [0, 1]], dtype=torch.bool)
        tags = torch.tensor([[0, 1], [-100, 0], [-100, 1]])
        clean = emissions.detach().clone()
        with torch.no_grad():
            emissions[~mask] = float("nan")
        expected = model(clean, tags, mask)
        actual = model(emissions, tags, mask.byte())
        torch.testing.assert_close(actual, expected)
        actual.neg().backward()
        self.assertTrue(torch.isfinite(emissions.grad).all())
        self.assertTrue((emissions.grad[~mask] == 0).all())
        torch.testing.assert_close(
            model.decode(clean, mask), model.decode(clean, mask.byte())
        )

    def test_gradcheck(self):
        model = CRF(2).double()
        emissions = torch.randn(3, 2, 2, dtype=torch.double, requires_grad=True)
        tags = torch.tensor([[0, 1], [1, 0], [0, 1]])
        mask = torch.tensor([[1, 1], [1, 1], [0, 1]], dtype=torch.bool)
        self.assertTrue(
            torch.autograd.gradcheck(lambda x: model(x, tags, mask), emissions)
        )

    def test_training_and_resume(self):
        model = CRF(2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
        emissions = torch.randn(3, 2, 2)
        tags = torch.tensor([[0, 1], [1, 0], [0, 1]])
        initial = copy.deepcopy(model.state_dict())

        def step(network, opt):
            network.train()
            opt.zero_grad(set_to_none=True)
            loss = -network(emissions, tags, reduction="token_mean")
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(
                all(torch.isfinite(p.grad).all() for p in network.parameters())
            )
            opt.step()

        step(model, optimizer)
        state = copy.deepcopy(model.state_dict())
        opt_state = copy.deepcopy(optimizer.state_dict())
        step(model, optimizer)
        resumed = CRF(2)
        resumed.load_state_dict(state)
        resumed_opt = torch.optim.Adam(resumed.parameters(), lr=0.03)
        resumed_opt.load_state_dict(opt_state)
        step(resumed, resumed_opt)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(
                value, resumed.state_dict()[name], rtol=0, atol=0
            )
        self.assertTrue(
            any(not torch.equal(initial[k], v) for k, v in model.state_dict().items())
        )
        model.eval()
        torch.testing.assert_close(model.decode(emissions), model.decode(emissions))

    def test_invalid_inputs(self):
        model = CRF(2)
        emissions = torch.randn(3, 2, 2)
        tags = torch.zeros(3, 2, dtype=torch.long)
        for nbest in (0, -1, 1.5, True, 9):
            with self.subTest(nbest=nbest), self.assertRaises(ValueError):
                model.decode(emissions, nbest=nbest)
        for mask in (torch.tensor([[1, 1], [0, 1], [1, 1]]), torch.zeros(3, 2)):
            with self.assertRaises(ValueError):
                model(emissions, tags, mask)
        for bad in (torch.empty(0, 2, 2), torch.empty(3, 0, 2), emissions.long()):
            with self.assertRaises(ValueError):
                model.decode(bad)
        with self.assertRaises(ValueError):
            model(emissions, tags.float())
        with self.assertRaises(ValueError):
            model(emissions, tags - 1)
        with self.assertRaises(ValueError):
            model.decode(emissions, torch.tensor([[1, 1], [0, 1], [0, 1]]), nbest=3)

    def test_mixed_dtype_likelihood_and_gradients(self):
        for parameter_dtype, emission_dtype in (
            (torch.float32, torch.float64),
            (torch.float16, torch.float32),
            (torch.bfloat16, torch.float32),
            (torch.float64, torch.float32),
        ):
            with self.subTest(parameters=parameter_dtype, emissions=emission_dtype):
                model = CRF(3).to(dtype=parameter_dtype)
                promoted = torch.promote_types(parameter_dtype, emission_dtype)
                reference = copy.deepcopy(model).to(dtype=promoted)
                emissions = torch.randn(
                    3, 2, 3, dtype=emission_dtype, requires_grad=True
                )
                reference_emissions = emissions.detach().to(promoted).requires_grad_()
                tags = torch.tensor([[0, 1], [1, 2], [-100, 0]])
                mask = torch.tensor([[1, 1], [1, 1], [0, 1]], dtype=torch.bool)
                actual = model(emissions, tags, mask, reduction="none")
                expected = reference(reference_emissions, tags, mask, reduction="none")
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                actual.sum().backward()
                expected.sum().backward()
                torch.testing.assert_close(
                    emissions.grad, reference_emissions.grad.to(emission_dtype)
                )
                for parameter, reference_parameter in zip(
                    model.parameters(), reference.parameters()
                ):
                    # Low-precision leaf gradients round each accumulated contribution.
                    tolerance = 2 * torch.finfo(parameter_dtype).eps
                    torch.testing.assert_close(
                        parameter.grad,
                        reference_parameter.grad.to(parameter_dtype),
                        rtol=tolerance,
                        atol=tolerance,
                    )

    def test_mixed_dtype_nbest_preserves_ranking(self):
        model = CRF(2).double()
        with torch.no_grad():
            model.start_transitions.copy_(
                torch.tensor([1.0, 1.0 + 1e-8], dtype=torch.double)
            )
            model.end_transitions.zero_()
            model.transitions.zero_()
        emissions = torch.zeros(1, 1, 2)
        expected = torch.tensor([[[1]], [[0]]])
        torch.testing.assert_close(model.decode(emissions, nbest=2), expected)
        torch.testing.assert_close(model.decode(emissions), expected[:1])

    def test_all_gradients_against_enumeration(self):
        model = CRF(2).double()
        emissions = torch.randn(3, 2, 2, dtype=torch.double, requires_grad=True)
        tags = torch.tensor([[0, 1], [1, 0], [-100, 1]])
        mask = torch.tensor([[1, 1], [1, 1], [0, 1]], dtype=torch.bool)
        expected = []
        for batch, length in enumerate((2, 3)):
            paths, scores = enumerate_paths(model, emissions[:length, batch])
            gold = paths.index(tuple(tags[:length, batch].tolist()))
            expected.append(scores[gold] - scores.logsumexp(0))
        inputs = (emissions, *model.parameters())
        oracle = torch.autograd.grad(torch.stack(expected).sum(), inputs)
        actual = torch.autograd.grad(model(emissions, tags, mask), inputs)
        for actual_gradient, expected_gradient in zip(actual, oracle):
            torch.testing.assert_close(actual_gradient, expected_gradient)

    def test_can_overfit_tiny_batch(self):
        model = CRF(3)
        # Trainable emissions stand in for an upstream tagger's output layer.
        emissions = torch.nn.Parameter(torch.randn(4, 2, 3))
        tags = torch.tensor([[0, 2], [1, 0], [2, -100], [1, -100]])
        mask = tags != -100
        optimizer = torch.optim.Adam([*model.parameters(), emissions], lr=0.1)
        initial = -model(emissions, tags, mask, reduction="token_mean").detach()
        model.train()
        for _ in range(80):
            optimizer.zero_grad(set_to_none=True)
            loss = -model(emissions, tags, mask, reduction="token_mean")
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            for parameter in (*model.parameters(), emissions):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            optimizer.step()
        self.assertLess(float(loss.detach()), float(initial) * 0.02)
        model.eval()
        decoded = model.decode(emissions)[0].transpose(0, 1)
        torch.testing.assert_close(decoded[mask], tags[mask])

    def test_state_dict_serialization(self):
        model = CRF(3, batch_first=True)
        emissions = torch.randn(2, 3, 3)
        tags = torch.randint(3, (2, 3))
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = CRF(3, batch_first=True)
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        torch.testing.assert_close(restored(emissions, tags), model(emissions, tags))
        torch.testing.assert_close(
            restored.decode(emissions, nbest=3), model.decode(emissions, nbest=3)
        )

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_mps_matches_cpu(self):
        for batch_first in (False, True):
            with self.subTest(batch_first=batch_first):
                cpu = CRF(3, batch_first=batch_first)
                mps = copy.deepcopy(cpu).to("mps")
                emissions = torch.randn(3, 2, 3)
                tags = torch.tensor([[0, 1], [1, 0], [-100, 2]])
                mask = tags != -100
                if batch_first:
                    emissions, tags, mask = (
                        x.transpose(0, 1) for x in (emissions, tags, mask)
                    )
                emissions = emissions.detach().requires_grad_()
                mps_emissions = emissions.detach().to("mps").requires_grad_()
                actual = mps(mps_emissions, tags.to("mps"), mask.to("mps"))
                expected = cpu(emissions, tags, mask)
                torch.testing.assert_close(actual.cpu(), expected)
                actual.backward()
                expected.backward()
                torch.testing.assert_close(mps_emissions.grad.cpu(), emissions.grad)
                for actual_parameter, expected_parameter in zip(
                    mps.parameters(), cpu.parameters()
                ):
                    torch.testing.assert_close(
                        actual_parameter.grad.cpu(), expected_parameter.grad
                    )
                for nbest in (1, 5):
                    torch.testing.assert_close(
                        mps.decode(mps_emissions, mask.to("mps"), nbest=nbest).cpu(),
                        cpu.decode(emissions, mask, nbest=nbest),
                    )

    def test_single_tag(self):
        model = CRF(1)
        emissions = torch.randn(4, 2, 1)
        torch.testing.assert_close(
            model.decode(emissions), torch.zeros(1, 2, 4, dtype=torch.long)
        )
        with self.assertRaises(ValueError):
            model.decode(emissions, nbest=2)


if __name__ == "__main__":
    unittest.main()
