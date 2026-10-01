# stateless-gradient-whitening

A small **stateless** matrix optimizer for Linux CPUs. Every parameter update
depends only on the current gradient and the configuration passed to that call —
no momentum, no second-moment accumulators, no hidden state.

## Algorithm

For parameters `P` and gradient `G` (real matrices with `1 <= rows <= cols <= 32`):

1. **Row RMS normalization** — each row of `G` is divided by its own root mean
   square. Zero rows skip the division and stay zero; an all-zero gradient
   therefore produces an exactly zero update.
2. **Whitening** — form the row Gram matrix, add the stability term to it only
   (`M = G_norm G_norm^T + eps I`), and compute the inverse square root
   `W = M^{-1/2}` via a Jacobi eigendecomposition. The whitened direction is
   `D = W G_norm`. Rank-deficient and nearly correlated rows stay finite
   because `eps > 0` keeps every eigenvalue positive; a diverging matrix is
   never silently swapped for the identity.
3. **Optional norm restore** — `D` is rescaled to `||G_norm||_F` when
   `restore_norm` is enabled (disable it to compare against an independent
   whitening reference).
4. **Update** — `P <- P - lr * D`. With `lr = 0` the direction is still
   computed and checked, but `P` is returned unchanged.

Each call returns the normalized gradient, the whitened direction, the updated
parameters, the whitening error `||W^2 M - I||_F / sqrt(rows)`, and the
stability epsilon actually used.

## Validation

The step is rejected (and the inputs are never modified) for:

- illegal shapes (empty, `rows > cols`, any dimension > 32, mismatched shapes),
- non-finite parameters or gradients,
- non-positive or non-finite `epsilon`,
- negative or non-finite `learning_rate`.

## Build, test, run

```sh
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
./build/quadratic_demo
```

`quadratic_demo` minimizes a fixed quadratic `f(P) = 0.5 * sum a_ij (P_ij -
P*_ij)^2` for 12 steps and prints the true objective value at every step; the
run is fully deterministic.

## Dependencies

Zero third-party dependencies — only a C++17 compiler, CMake (>= 3.16) and
CTest, all of which work fully offline. No external services and no GPU are
required. (The design allows up to two third-party libraries; none were
needed. If you later vendor a local matrix library, drop its sources into
`third_party/` so offline builds keep working, and document the version here.)

## Layout

- `include/stateless_whitener.h` — public API (`whiten_step`, config, result)
- `src/stateless_whitener.cpp` — implementation
- `examples/quadratic_demo.cpp` — reproducible multi-step quadratic example
- `tests/test_stateless_whitener.cpp` — self-contained test runner (CTest entry)
