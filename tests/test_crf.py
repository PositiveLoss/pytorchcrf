import copy
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
            for lengths, nbest in (([1, 1], 3), ([1, 3], 3), ([2, 3], 7), ([3, 3], 1)):
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
