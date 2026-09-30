# stateless-matrix-whitener

A small **stateless** matrix optimizer for Linux/CPU, in C++17 with **zero
third-party dependencies** (standard library only; no GPU, no network).

Every parameter update depends only on the **current gradient** and the
**configuration of this single call**. The optimizer keeps no momentum buffer,
no running average and no second-order statistics, so two calls given the same
`(gradient, config)` produce bit-identical directions regardless of history.

## What one step does

Given gradient `G` and parameters `P0` (both `m x n`, `1 <= m <= n <= 32`):

1. **Per-row RMS normalization** `N_ij = G_ij / rms(row_i)`.
   An all-zero row skips the division and stays exactly zero; a non-zero row
   uses its own RMS directly.
2. **Row Gram matrix** `Gr = N N^T` (`m x m`).
3. **Regularization on the Gram matrix only** `A = Gr + reg * I`
   (`reg > 0`). The normalized gradient itself is never changed by `reg`.
4. **Whitening** `W = A^{-1/2}` via a symmetric Jacobi eigendecomposition.
   Rank-deficient / near-correlated rows stay finite because deficient
   directions map to `1/sqrt(reg)`. The inverse square root is genuinely
   computed — it is **never silently replaced by the identity matrix**.
5. **Direction** `D = W N`. With `restore_overall_scale = true`, `D` is
   rescaled so that `||D||_F = ||N||_F`; with it off, the raw whitened product
   is used.
6. **Update** `P = P0 - learning_rate * D`.

The result returns the normalized gradient, whitened direction, updated
parameters, the whitener `W`, the **whitening error** (relative backward error
`||W^2 - A^{-1}||_F / ||A^{-1}||_F`, which is not amplified by the condition
number) and the **stability parameter actually used**.

## Inputs that are rejected (parameters never modified)

- empty / illegal shapes, `m > n`, or any dimension `> 32`,
- non-finite gradient or parameters (`NaN`, `Inf`),
- non-positive or non-finite `regularization`,
- negative or non-finite `learning_rate`.

`learning_rate = 0` is accepted: the direction is still computed and returned,
but the parameters are returned unchanged.

## Build and test

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

CTest entries:

- `unit_tests` — 147 assertions across 11 test groups (see below).
- `quadratic_demo_converges` — the reproducible multi-step quadratic demo.
- `reject_*` — six **negative** tests marked `WILL_FAIL`. They feed invalid
  configuration (rows > columns, dimension 33, **zero regularization zero
  case**, negative regularization, negative learning rate, NaN gradient) and
  must exit non-zero. This is the "test zero cases should fail" entry: the
  zero-stability-parameter case is asserted to be rejected.

## Reproducible quadratic demo

`quadratic_demo` minimizes the separable quadratic

```
f(P) = 0.5 * sum_ij w_ij (P_ij - T_ij)^2,   G_ij = w_ij (P_ij - T_ij)
```

where `T` (target) and `W` (positive weights) are generated deterministically
from `(m, n)`. The true optimum is `P* = T` and the true minimum is exactly
`f* = 0`, both known and printed. It starts at `P0 = 0` and runs a
deterministic geometric learning-rate schedule (the rate is part of each
stateless call's config; it is not stored by the optimizer).

```sh
./build/quadratic_demo                       # defaults: 2x3, reg 1e-8
./build/quadratic_demo --no-restore
./build/quadratic_demo 1e-8 0.2 120 --m 4 --n 8
./build/quadratic_demo 0.0 0.1              # exit code 2: rejected
./build/quadratic_demo --nan                # exit code 2: rejected
```

Default run reaches `f(P) ≈ 2.5e-10` after 120 steps.

## Test coverage

`tests/test_optimizer.cpp` covers:

- **real update values** — a hand-computed 1x2 step (normalized gradient,
  whitened direction, updated parameters),
- **row scaling invariance** — positive per-row scaling leaves N/D/P
  unchanged; negative scaling only flips that row's sign,
- **independent whitening reference** — the test computes `A^{-1/2}` itself
  with a different algorithm (scaled Newton iteration on `Y <- 0.5 Y (3I -
  B Y^2)`, in `tests/independent_reference.hpp`) and compares the whitener and
  directions to it with overall scaling **off**, within `1e-9`,
- **rank-deficient and near-correlated rows** — finite results, large
  `1/sqrt(reg)` factors (proving no identity substitution), small backward
  error,
- **zero gradient and zero rows** — zero update, exact zero rows preserved,
- **no historical state** — bit-identical output after interleaving unrelated
  calls,
- **rejected inputs leave parameters unchanged** — bytewise comparison
  including `NaN` payloads,
- zero learning rate, the overall-scale switch, 32x32 max size, and the 33x33
  boundary.

## Offline / dependency notes

- Third-party dependencies: **0** (well under the two-dependency limit).
- No matrix library needs installing; everything is local under
  `include/` and `src/` (`smw::Matrix` plus a self-contained cyclic Jacobi
  eigensolver). To build offline, only a C++17 compiler and CMake >= 3.16 are
  required. Nothing is downloaded at configure or build time.
- Linux CPU only; no external services and no GPU code paths.
