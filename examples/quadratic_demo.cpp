// Deterministic multi-step demo on a small quadratic objective.
//
//   f(P) = 0.5 * sum_ij a_ij * (P_ij - P*_ij)^2,   a_ij = 1 + i + j/10
//
// Gradient: dF/dP_ij = a_ij * (P_ij - P*_ij). Everything is computed with
// closed-form formulas, so runs are bit-for-bit reproducible.
#include <cmath>
#include <cstdio>

#include "stateless_whitener.h"

namespace {

constexpr std::size_t kRows = 3;
constexpr std::size_t kCols = 5;
constexpr int kSteps = 12;

double curvature(std::size_t i, std::size_t j) {
    return 1.0 + static_cast<double>(i) + 0.1 * static_cast<double>(j);
}

double optimum(std::size_t i, std::size_t j) {
    return 0.05 * static_cast<double>(i) - 0.03 * static_cast<double>(j);
}

double objective(const sw::Matrix& p) {
    double f = 0.0;
    for (std::size_t i = 0; i < p.rows; ++i)
        for (std::size_t j = 0; j < p.cols; ++j) {
            const double d = p.at(i, j) - optimum(i, j);
            f += 0.5 * curvature(i, j) * d * d;
        }
    return f;
}

sw::Matrix gradient(const sw::Matrix& p) {
    sw::Matrix g(p.rows, p.cols);
    for (std::size_t i = 0; i < p.rows; ++i)
        for (std::size_t j = 0; j < p.cols; ++j)
            g.at(i, j) = curvature(i, j) * (p.at(i, j) - optimum(i, j));
    return g;
}

}  // namespace

int main() {
    sw::Matrix params(kRows, kCols);
    for (std::size_t i = 0; i < kRows; ++i)
        for (std::size_t j = 0; j < kCols; ++j)
            params.at(i, j) = 0.7 + 0.11 * static_cast<double>(i * kCols + j);

    sw::WhitenConfig config;
    config.learning_rate = 0.05;
    config.epsilon = 1e-3;
    config.restore_norm = true;

    std::printf("stateless whitening on a %zux%zu quadratic (lr=%.3f, eps=%.1e)\n", kRows,
                kCols, config.learning_rate, config.epsilon);
    std::printf("%4s  %16s  %16s  %14s\n", "step", "objective", "whiten_error", "grad_norm");

    for (int step = 0; step <= kSteps; ++step) {
        const sw::Matrix grad = gradient(params);
        double grad_norm = 0.0;
        for (double v : grad.data) grad_norm += v * v;
        grad_norm = std::sqrt(grad_norm);

        const sw::WhitenResult r = sw::whiten_step(params, grad, config);
        if (!r.ok) {
            std::fprintf(stderr, "step %d rejected: %s\n", step, r.error.c_str());
            return 1;
        }
        std::printf("%4d  %16.10f  %16.3e  %14.6e\n", step, objective(params),
                    r.whitening_error, grad_norm);
        params = r.updated_params;  // optimizer itself keeps no state; we carry params
    }
    return 0;
}
