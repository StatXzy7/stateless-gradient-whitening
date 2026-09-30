// Stateless per-row RMS-normalized gradient whitening optimizer.
//
// Given the current gradient G (m x n, m <= n, each dimension <= 32) and the
// current parameters P0 of the same shape, one step performs:
//
//   1. N  = row-wise RMS-normalized G (all-zero rows stay exactly zero).
//   2. Gr = N * N^T                                    (m x m row Gramian).
//   3. A  = Gr + reg * I                               (regularization added
//                                                         only to the Gramian).
//   4. W  = A^{-1/2}                                   (real inverse square
//                                                         root via eigendecomp).
//   5. D  = W * N, optionally rescaled so that ||D||_F == ||N||_F.
//   6. P  = P0 - learning_rate * D.
//
// The optimizer stores no momentum and no second-order statistics: every call
// depends only on (G, P0, Config). Repeated calls with identical (G, Config)
// return bit-identical directions.
#pragma once

#include <cstddef>
#include <string>

#include "smw/matrix.hpp"

namespace smw {

inline constexpr std::size_t kMaxDim = 32;
inline constexpr double kDefaultRegularization = 1e-8;
inline constexpr double kDefaultLearningRate = 0.1;

struct Config {
    // Positive stability parameter added to the diagonal of the row Gramian.
    double regularization = kDefaultRegularization;
    // Non-negative step size applied to the whitened direction.
    double learning_rate = kDefaultLearningRate;
    // When true, the whitened direction is rescaled to the total norm of the
    // normalized gradient; when false, the raw whitened direction is used.
    bool restore_overall_scale = true;
};

struct Result {
    Matrix normalized_gradient;  // N
    Matrix whitened_direction;   // D
    Matrix updated_parameters;   // P
    Matrix whitener;             // W = (N N^T + reg I)^{-1/2}
    // Relative backward error ||W^2 - A^{-1}||_F / ||A^{-1}||_F, which is
    // not amplified by the condition number of A. (The forward residual
    // ||W A W - I|| is naturally large in ill-conditioned deficient
    // directions even for an accurate inverse square root.)
    double whitening_error = 0.0;
    double used_regularization = 0.0;
    bool accepted = false;
    std::string reject_reason;
};

// Validates shapes and finiteness without touching the optimizer state
// (there is no state). Empty, wrong-shape, non-finite inputs, non-positive
// regularization and negative learning rates are rejected.
bool validate_step_inputs(const Matrix &gradient,
                          const Matrix &parameters,
                          const Config &config,
                          std::string *reason = nullptr);

// Performs one stateless update. On rejection, `out` is returned with
// accepted == false and a reject_reason, and the caller's parameters matrix
// is never modified (parameters is passed by const reference).
Result step(const Matrix &gradient,
            const Matrix &parameters,
            const Config &config);

}  // namespace smw
