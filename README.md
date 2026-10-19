# PyTorch CRF with N-best decoding

A linear-chain conditional random field (CRF) for sequence tagging in PyTorch.
Use it on top of a tagger's emission scores to learn tag transitions, compute
sequence log likelihoods, and decode the most likely tag sequences.

This project is a fork of [pytorch-crf](https://github.com/kmkurn/pytorch-crf),
originally authored by [Kemal Kurniawan](https://github.com/kmkurn).

- Best-path and N-best Viterbi decoding.
- Variable-length batches with padding masks.
- Sequence-first and batch-first tensor layouts.
- Autograd support for training and TorchScript support for serialization.

## Installation

Requires **Python >= 3.14** and **PyTorch >= 2.14.1**.

Install this repository's `main` branch directly from GitHub (requires Git):

```bash
python -m pip install "git+https://github.com/PositiveLoss/pytorchcrf.git@main"
```

To install a specific revision, replace `main` with a commit hash or tag.

The package is also installable by name:

```bash
python -m pip install pytorchcrf
```

Use the GitHub command for the implementation documented here; a published
package may differ from this repository.

## Quick start

This example trains a small tagger and CRF together. Emissions are raw scores
for each tag; no softmax is needed.

```python
import torch
from pytorchcrf import CRF

torch.manual_seed(17)
num_tags = 5
crf = CRF(num_tags, batch_first=True)
tagger = torch.nn.Linear(8, num_tags)
optimizer = torch.optim.Adam([*tagger.parameters(), *crf.parameters()], lr=0.01)

features = torch.randn(2, 4, 8)  # (batch_size, sequence_length, feature_size)
tags = torch.tensor([[0, 2, -100, -100], [1, 3, 4, 0]], dtype=torch.long)
mask = torch.tensor([[True, True, False, False], [True, True, True, True]])

tagger.train()
crf.train()
optimizer.zero_grad(set_to_none=True)
emissions = tagger(features)  # (batch_size, sequence_length, num_tags)
loss = -crf(emissions, tags, mask, reduction="token_mean")
loss.backward()
optimizer.step()
```

`forward` returns a **log likelihood**. Minimize its negative when training.
Include both the tagger and CRF parameters in your optimizer.

### Decode tag sequences

Using the models and inputs above:

```python
tagger.eval()
crf.eval()
with torch.no_grad():
    emissions = tagger(features)

best = crf.decode(emissions, mask, pad_tag=-1)            # shape: (1, 2, 4)
top3 = crf.decode(emissions, mask, nbest=3, pad_tag=-1)   # shape: (3, 2, 4)
best_per_sample = best[0]                               # shape: (2, 4)
```

`decode` returns a `torch.long` tensor on the emissions device with shape
`(nbest, batch_size, sequence_length)`, regardless of the input layout. The rank
dimension is retained when `nbest=1`, the default. Paths are sorted by descending
score; the order of tied paths is unspecified. Decoding disables autograd.

`nbest` must be a positive integer no greater than
`num_tags ** shortest_sequence_length`, so every sample has enough distinct
paths. Padded output positions contain `pad_tag`, which defaults to `0`.

## Input contract

`CRF(num_tags)` defaults to sequence-first inputs. Set `batch_first=True` for
the layout used in the quick start.

| Tensor | Default layout | With `batch_first=True` | Dtype |
| --- | --- | --- | --- |
| `emissions` | `(sequence_length, batch_size, num_tags)` | `(batch_size, sequence_length, num_tags)` | Floating point |
| `tags` | `(sequence_length, batch_size)` | `(batch_size, sequence_length)` | `torch.long` |
| `mask` | `(sequence_length, batch_size)` | `(batch_size, sequence_length)` | `torch.bool` preferred; byte masks accepted |

- Each sequence must have at least one token, and the batch must be nonempty.
- `True` marks a valid token. Padding must occur only at the end of a sequence.
- Omitting `mask` treats every position as valid.
- Valid tags must be in `[0, num_tags)`. Masked tags may use a sentinel such as `-100`.
- Inputs and CRF parameters must be on the same device.

Likelihood scores and decoding preserve PyTorch dtype promotion when emissions
and parameters use different dtypes. Gradients stored in low-precision tensors
remain subject to rounding in that dtype.

### Loss reductions

Pass `reduction` to `crf(emissions, tags, mask, reduction=...)`:

| Reduction | Result | Shape |
| --- | --- | --- |
| `"none"` | One log likelihood per sequence | `(batch_size,)` |
| `"sum"` (default) | Sum of sequence log likelihoods | Scalar |
| `"mean"` | Sum divided by the batch size | Scalar |
| `"token_mean"` | Sum divided by the number of valid tokens | Scalar |

## TorchScript

Use `torch.jit.script` to preserve input-dependent loops and validation.
Both `forward` and `decode` remain available after saving and loading:

```python
import torch
from pytorchcrf import CRF

model = CRF(num_tags=5, batch_first=True).eval()
scripted = torch.jit.script(model)
torch.jit.save(scripted, "crf.pt")
restored = torch.jit.load("crf.pt")

emissions = torch.randn(2, 4, 5)
tags = torch.randint(5, (2, 4))
mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
log_likelihood = restored(emissions, tags, mask, reduction="token_mean")
paths = restored.decode(emissions, mask, nbest=3, pad_tag=-1)
```

Scripted modules accept different batch sizes, sequence lengths, and `nbest`
values at runtime. `forward` retains autograd support; `decode` disables it.

Scripting, save/load, output parity, and gradients were verified on CPU with
Python 3.14 and PyTorch 2.14.1. **PyTorch warns that TorchScript is unsupported
on Python 3.14+ and may break.** Passing checks do not remove that upstream
compatibility limitation. TorchScript execution on CUDA/MPS and C++ consumers
has not been validated.

## Development and tests

From a local checkout, install the package and development tools:

```bash
python -m pip install -e . "ruff>=0.17.1"
python -m unittest discover -s tests -v
ruff check src tests
ruff format --check src tests
```

The suite checks likelihoods and decoded paths against exhaustive enumeration,
gradients, padding, mixed dtypes, tiny-batch learning, checkpoint round trips,
and TorchScript parity. CPU/MPS parity is checked when MPS is available.

## Credits and license

The original CRF implementation is from
[pytorch-crf](https://github.com/kmkurn/pytorch-crf) by
[Kemal Kurniawan](https://github.com/kmkurn) and its contributors.
This fork builds on that work with batched Viterbi and N-best decoding
contributions by Feiyang Niu and maintenance by Yehor Smoliakov.

Released under the [MIT license](LICENSE), retaining the original author's
copyright notice alongside those of this fork's contributors.
