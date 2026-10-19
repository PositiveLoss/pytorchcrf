# PyTorch CRF with N-best Decoding

Implementation of Conditional Random Fields (CRF) in PyTorch. It supports top-N most probable paths decoding.

The package is based on [pytorch-crf](https://github.com/kmkurn/pytorch-crf) with only the following differences

- Method `_viterbi_decode` that decodes the most probable path get optimized. Running time gets reduced to 50% or less with batch size 15+ and sequence length 20+
- The class now supports decoding top-N most probable paths through the implementation of the method `_viterbi_decode_nbest`

## Requirements

- Python >= 3.14
- PyTorch >= 2.14.1

## Installation

```bash
pip install pytorchcrf
```

## Examples

```python
>>> import torch
>>> from pytorchcrf import CRF
>>> num_tags = 5                        # number of tags is 5
>>> model = CRF(num_tags)
>>> seq_length = 3                      # maximum sequence length in a batch
>>> batch_size = 2                      # number of samples in the batch
>>> emissions = torch.randn(seq_length, batch_size, num_tags)

# Computing log likelihood
>>> tags = torch.tensor([[2, 3], [1, 0], [3, 4]], dtype=torch.long)  # (seq_length, batch_size)
>>> model(emissions, tags)

# Decoding
>>> model.decode(emissions)             # decoding the best path
>>> model.decode(emissions, nbest=3)    # decoding the top 3 paths
```

`decode` returns a `torch.long` tensor on the emissions device with shape
`(nbest, batch_size, seq_length)`, including when `nbest=1`. Paths are ordered by
descending score; ordering among tied scores is unspecified. `nbest` must be a
positive integer no greater than `num_tags ** shortest_sequence_length`.
Padded positions contain `pad_tag` (default `0`). Decoding disables autograd.

For variable lengths, supply a boolean mask matching the first two emission
dimensions. Byte masks are also accepted. Each sequence must contain at least
one token, and padding must occur only at the end. Tags must have dtype
`torch.long`; unmasked tags must be in `[0, num_tags)`. Masked tags may contain
padding values such as `-100`. Inputs and CRF parameters must be on the same device.

For training, minimize the negative log likelihood, for example
`loss = -model(emissions, tags, mask, reduction="token_mean")`. The `mean`
reduction averages over sequences; `token_mean` divides the summed log likelihood
by the number of unmasked tokens.

Run the correctness and training checks with:

```bash
python -m unittest discover -s tests -v
```
