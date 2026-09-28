# aiayn: "Attention Is All You Need" in C++23

A from-scratch, dependency-free implementation of the Transformer of Vaswani et al. (2017,
[arXiv:1706.03762](https://arxiv.org/abs/1706.03762)): the full **encoder-decoder** model, **training** (reverse-mode autodiff, Adam,
the paper's learning-rate schedule, label smoothing, dropout) and **inference** (beam search with length penalty).
Header-only library (~1,080 lines), a training CLI, and ~760 lines of tests.

It is a *reference* implementation: small, readable and heavily verified, not fast. It has not been trained on WMT (see "Scope").

## Paper to code

| Paper | Where | Notes |
|---|---|---|
| 3.1 Encoder / decoder stacks, `LayerNorm(x + Sublayer(x))` | `model.hpp` `sublayer`, `encode`, `decode` | post-LN, N layers each |
| 3.2.1 Scaled dot-product attention, eq. (1) | `model.hpp` `mha` | `softmax(QK^T / sqrt(d_k)) V` |
| 3.2.2 Multi-head attention | `model.hpp` `mha` | `W_i^Q, W_i^K, W_i^V` are column blocks of one `[d_model x d_model]` matrix; `W^O` merges the heads |
| 3.2.3 Encoder-decoder attention, masked decoder self-attention | `masked_softmax` in `autograd.hpp` | causal mask and source-padding mask; masked entries have weight exactly 0 (= -inf before the softmax) |
| 3.3 Position-wise FFN, eq. (2) | `model.hpp` `ffn` | `max(0, x W1 + b1) W2 + b2` |
| 3.4 Embeddings and softmax | `model.hpp` `embed`, `decode` | shared source/target embeddings (optional, needs one vocabulary), pre-softmax projection tied to the target embedding, entries scaled by `sqrt(d_model)` |
| 3.5 Sinusoidal positional encoding | `positional_encoding` | `PE(pos,2i)=sin(pos/10000^(2i/d))`, `PE(pos,2i+1)=cos(...)` |
| 5.3 Adam and the schedule, eq. (3) | `optim.hpp` | beta1 0.9, beta2 0.98, eps 1e-9, `d^-0.5 * min(step^-0.5, step * warmup^-1.5)` |
| 5.4 Residual dropout, label smoothing | `dropout`, `cross_entropy` | dropout on sub-layer outputs and on embeddings+PE (and, like the paper, **not** on attention weights); eps_ls = 0.1 |
| 6.1 Beam search, alpha = 0.6, max length input + 50, early termination | `decode.hpp` | GNMT length penalty `((5 + len) / 6)^alpha` |
| Table 3 model sizes | `Hyper::base`, `Hyper::big` | see "Model sizes" |

Choices the paper does not specify: LayerNorm epsilon 1e-6; Xavier-uniform initialization for weight matrices; embeddings
initialized `N(0, d_model^-1/2)`; label smoothing puts `1-eps` on the gold token and `eps/(V-1)` on each other token; attention
projections have no bias, the FFN has biases (as in the equations).

## Build and run

```bash
cmake -S . -B build && cmake --build build -j
ctest --test-dir build --output-on-failure          # float + double test suites (~20 s)

./build/train_toy --task reverse --steps 2000 --warmup 400 --lr-factor 0.5   # ~1 min on one core
./build/train_toy --task sort    --steps 2000 --warmup 400 --lr-factor 0.5
./build/train_toy --paper-sizes                                               # parameter counts of base and big
```

Requires a C++23 compiler. `-DAIAYN_NATIVE=OFF` gives a portable binary. Precision is a compile-time switch:
`-DAIAYN_REAL=double`. Builds pass `-ffp-contract=off` so floating-point results are reproducible across
platforms regardless of `-march=native` (FMA contraction would otherwise perturb the `float` build).

### Using the library

```cpp
#include "aiayn/tasks.hpp"
using namespace aiayn;

Hyper h = Hyper::base(/*shared vocab*/ 37000);      // or fill in the fields yourself
Transformer model(h, /*seed=*/1);
Adam opt;                                           // paper defaults
std::mt19937_64 rng(1);

// training: one optimizer step on a mini-batch of {src, tgt} token sequences
double loss = train_step(model, opt, batch, noam_lr(step, h.d_model, 4000), rng);

// inference: beam size 4, alpha 0.6
DecodeResult r = beam_search(model, src, (int)src.size(), BeamOptions{});
```

## Results

Toy sequence tasks (`--task copy|reverse|sort`, alphabet 12, lengths 3-8), N=2, d_model=64, 4 heads, d_ff=128, with the
paper's regularization (dropout 0.1, label smoothing 0.1). Exact-match accuracy on 300 unseen examples after 2000 steps of
32 examples, about one minute on one CPU core, with the command shown above (`--warmup 400 --lr-factor 0.5`), for seeds 1, 2, 3
(greedy and beam are scored on the same 300 examples):

| Task | Greedy (per seed) | Beam 4, alpha 0.6 (per seed) |
|---|---|---|
| reverse | 99.0 / 92.7 / 96.0 % (mean 95.9) | 99.0 / 92.7 / 96.3 % (mean 96.0) |
| sort | 93.7 / 94.7 / 96.0 % (mean 94.8) | 94.0 / 95.0 / 96.3 % (mean 95.1) |

Copy reaches 100 % in about 1000 steps with a smaller model (used by the test-suite). Seeds differ by up to 6 points, and
beam search is at most 0.3 points better than greedy here: with three seeds and 300 examples, that difference is noise, not
evidence. The training loss plateaus near 0.6 rather than 0: that is the entropy of the label-smoothed target, which the paper
notes hurts perplexity while helping accuracy.

**Learning rate matters for small models.** The paper's schedule peaks at `d_model^-0.5 * warmup^-0.5` (about 7e-4 for
d_model 512 and 4000 warmup steps). A tiny model with a short warmup can peak 50x higher and simply fails to learn: with a peak
of 0.035 the copy task stayed at chance, with a peak of 0.003-0.006 it reached 100 %. Use `--warmup` and `--lr-factor`
accordingly.

### Model sizes (vocabulary of 37,000 shared tokens)

| Model | This implementation | Paper, table 3 |
|---|---|---|
| base (N=6, d_model=512, d_ff=2048, h=8) | 63.0 M | 65 M |
| big (N=6, d_model=1024, d_ff=4096, h=16) | 214.2 M | 213 M |

Big matches; base is 3 % below the table and I do not have a verified explanation for the gap.

The paper's base architecture runs end to end here: three real training steps at N=6, d_model=512, d_ff=2048, h=8 (vocabulary
reduced to 1,000, 44.6 M parameters) take about 1.2 s each for two 8-token examples on one core, starting from a loss near
`ln V`.

## How it is verified

`tests/test_all.cpp` (~6,600 checks, ~760 lines) is built twice, in `float` and in `double`:

* **Independent reference.** A naive double-precision implementation written directly from the equations of section 3, sharing
  no code with the library, must produce the same logits (1-3 layers, 1-4 heads, shared and separate embeddings,
  padded sources).
* **Gradients.** Reverse-mode gradients of the *whole model* are compared with central finite differences (double build):
  tied and separate embeddings, label smoothing on and off, padded source and target, and dropout with fixed masks. Every
  parameter matrix must receive a non-zero gradient.
* **Properties.** Decoder causality; padding content is irrelevant and equivalent to no padding; sinusoidal encodings satisfy the
  paper's "PE(pos+k) is a linear function of PE(pos)" hypothesis exactly; layer norm, masked softmax, label-smoothed
  cross-entropy checked against their definitions.
* **Optimizer.** Adam against an independent scalar implementation, and equation (3) (peak position, linear warmup, inverse-sqrt decay).
* **Decoding.** Beam size 1 equals greedy; a beam wider than the search space equals *exhaustive search*, over 12 random models
  and alpha in {0, 0.6, 1.5, 2.5}, which also validates the early-termination rule.
* **Learning.** Copy and reverse are solved to > 90 % on unseen inputs, and training is bit-for-bit deterministic for a given seed.
* **Checkpoints and validation.** Round trip, truncated and foreign files rejected, invalid hyper-parameters and inputs throw.

The tests themselves were validated by mutation: `python3 tools/mutation_check.py` injects 19 classic bugs one at a time
(missing `1/sqrt(d_k)`, missing embedding scale, dropped residual, non-causal decoder, ignored padding mask, wrong softmax or
layer-norm backward, wrong positional encoding, wrong label smoothing, Adam without bias correction, wrong schedule exponent,
untied output gradient, missing ReLU, over-eager beam termination, wrong length penalty, unscaled dropout, swapped cross-attention
operands...) and all 19 are detected. The first run missed one (a too-eager early-termination bound in beam search); the test
was strengthened until it was caught.

## Scope and limitations

* **Not a reproduction of the paper's BLEU scores.** That needs WMT 2014 data, a 37k byte-pair vocabulary, batches of ~25k
  tokens, checkpoint averaging and 8 GPUs for 12 hours. None of that is included; the data pipeline is three toy tasks.
* CPU only, single-threaded, naive GEMM. A training step of the base model takes seconds, not milliseconds.
* Mini-batches are processed one example per tape with gradients accumulated (no padded batches), which is why training needs
  no padding. The padding mask is implemented and tested for the encoder/cross-attention, ready for a batched pipeline.
* Decoding recomputes the decoder over the whole prefix at each step (no key/value cache): correct and simple, O(n^2) per sequence.
* No gradient clipping, checkpoint averaging or learned-position variant (table 3, row E).
