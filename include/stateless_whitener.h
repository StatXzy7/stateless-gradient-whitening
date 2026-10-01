// Stateless RMS-normalize + Gram inverse-square-root whitening optimizer.
//
// Every update depends only on the current gradient and the configuration
// passed to this call. No momentum, no second-moment accumulators, no hidden
// state of any kind.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace sw {

// Row-major real matrix. Valid shapes: 1 <= rows <= cols <= 32.
struct Matrix {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<double> data;  // rows * cols, row-major

    Matrix() = default;
    Matrix(std::size_t r, std::size_t c) : rows(r), cols(c), data(r * c, 0.0) {}
    Matrix(std::size_t r, std::size_t c, std::vector<double> values);

    double& at(std::size_t i, std::size_t j) { return data[i * cols + j]; }
    double at(std::size_t i, std::size_t j) const { return data[i * cols + j]; }
};

struct WhitenConfig {
    double learning_rate = 1e-2;   // >= 0 required
    double epsilon = 1e-3;         // > 0 required; added to the Gram matrix only
    bool restore_norm = true;      // rescale whitened direction to ||normalized_grad||_F
};

struct WhitenResult {
    bool ok = false;
    std::string error;             // human-readable reason when ok == false

    Matrix normalized_grad;        // per-row RMS normalized gradient
    Matrix whitened_direction;     // (G G^T + eps I)^{-1/2} G, optionally norm-restored
    Matrix updated_params;         // params - lr * whitened_direction

    double whitening_error = 0.0;  // ||W^2 (G G^T + eps I) - I||_F / sqrt(rows)
    double stability_epsilon = 0.0;
};

// Validates inputs, then computes one stateless update step.
// On any validation failure returns ok == false and never touches `params`.
WhitenResult whiten_step(const Matrix& params, const Matrix& grad, const WhitenConfig& config);

}  // namespace sw
