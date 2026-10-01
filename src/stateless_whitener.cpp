#include "stateless_whitener.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sw {

namespace {

constexpr std::size_t kMaxDim = 32;

bool is_finite(const Matrix& m) {
    for (double v : m.data) {
        if (!std::isfinite(v)) return false;
    }
    return true;
}

bool shape_ok(const Matrix& m) {
    return m.rows >= 1 && m.cols >= 1 && m.rows <= m.cols && m.cols <= kMaxDim &&
           m.data.size() == m.rows * m.cols;
}

WhitenResult fail(std::string message) {
    WhitenResult r;
    r.ok = false;
    r.error = std::move(message);
    return r;
}

// Per-row RMS normalization. Zero rows skip the division and stay zero.
Matrix rms_normalize(const Matrix& g) {
    Matrix out(g.rows, g.cols);
    for (std::size_t i = 0; i < g.rows; ++i) {
        double sum_sq = 0.0;
        for (std::size_t j = 0; j < g.cols; ++j) sum_sq += g.at(i, j) * g.at(i, j);
        const double rms = std::sqrt(sum_sq / static_cast<double>(g.cols));
        if (rms == 0.0) continue;  // keep the row exactly zero
        const double inv = 1.0 / rms;
        for (std::size_t j = 0; j < g.cols; ++j) out.at(i, j) = g.at(i, j) * inv;
    }
    return out;
}

// A (n x m) -> A A^T (n x n).
std::vector<double> gram(const Matrix& a) {
    const std::size_t n = a.rows;
    std::vector<double> g(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < a.cols; ++k) s += a.at(i, k) * a.at(j, k);
            g[i * n + j] = s;
            g[j * n + i] = s;
        }
    return g;
}

// Jacobi eigenvalue decomposition of a symmetric n x n matrix.
// Returns eigenvalues and fills `vectors` (row-major, column j = eigenvector j).
std::vector<double> jacobi_eigh(std::vector<double> a, std::size_t n,
                                std::vector<double>& vectors) {
    vectors.assign(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) vectors[i * n + i] = 1.0;

    constexpr int kMaxSweeps = 100;
    const double tol = 1e-14;
    for (int sweep = 0; sweep < kMaxSweeps; ++sweep) {
        double off = 0.0;
        for (std::size_t p = 0; p < n; ++p)
            for (std::size_t q = p + 1; q < n; ++q) off += a[p * n + q] * a[p * n + q];
        if (std::sqrt(off) <= tol) break;

        for (std::size_t p = 0; p < n; ++p) {
            for (std::size_t q = p + 1; q < n; ++q) {
                const double apq = a[p * n + q];
                if (std::abs(apq) < 1e-300) continue;
                const double app = a[p * n + p];
                const double aqq = a[q * n + q];
                const double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
                const double c = std::cos(phi);
                const double s = std::sin(phi);
                for (std::size_t k = 0; k < n; ++k) {
                    const double akp = a[k * n + p];
                    const double akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for (std::size_t k = 0; k < n; ++k) {
                    const double apk = a[p * n + k];
                    const double aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for (std::size_t k = 0; k < n; ++k) {
                    const double vkp = vectors[k * n + p];
                    const double vkq = vectors[k * n + q];
                    vectors[k * n + p] = c * vkp - s * vkq;
                    vectors[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    std::vector<double> eigenvalues(n);
    for (std::size_t i = 0; i < n; ++i) eigenvalues[i] = a[i * n + i];
    return eigenvalues;
}

// W = V diag(1/sqrt(max(lambda_i, tiny))) V^T for symmetric PSD input.
std::vector<double> inverse_square_root(const std::vector<double>& eigenvalues,
                                        const std::vector<double>& vectors, std::size_t n) {
    const double tiny = std::numeric_limits<double>::min();
    std::vector<double> w(n * n, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        const double inv = 1.0 / std::sqrt(std::max(eigenvalues[j], tiny));
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t c = 0; c < n; ++c)
                w[r * n + c] += vectors[r * n + j] * inv * vectors[c * n + j];
    }
    return w;
}

// C = A (n x n) * B (n x m).
Matrix matmul(const std::vector<double>& a, std::size_t n, const Matrix& b) {
    Matrix c(n, b.cols);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < b.cols; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < n; ++k) s += a[i * n + k] * b.at(k, j);
            c.at(i, j) = s;
        }
    return c;
}

double frobenius_norm(const Matrix& m) {
    double s = 0.0;
    for (double v : m.data) s += v * v;
    return std::sqrt(s);
}

// ||W^2 M - I||_F / sqrt(n) for symmetric n x n W and M.
double whitening_error(const std::vector<double>& w, const std::vector<double>& m,
                       std::size_t n) {
    double acc = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                double w2_ik = 0.0;
                for (std::size_t l = 0; l < n; ++l)
                    w2_ik += w[i * n + l] * w[k * n + l];  // W symmetric
                s += w2_ik * m[k * n + j];
            }
            const double target = (i == j) ? 1.0 : 0.0;
            const double d = s - target;
            acc += d * d;
        }
    return std::sqrt(acc) / std::sqrt(static_cast<double>(n));
}

}  // namespace

Matrix::Matrix(std::size_t r, std::size_t c, std::vector<double> values)
    : rows(r), cols(c), data(std::move(values)) {}

WhitenResult whiten_step(const Matrix& params, const Matrix& grad, const WhitenConfig& config) {
    if (!shape_ok(params)) return fail("invalid params shape (need 1 <= rows <= cols <= 32)");
    if (!shape_ok(grad)) return fail("invalid grad shape (need 1 <= rows <= cols <= 32)");
    if (params.rows != grad.rows || params.cols != grad.cols)
        return fail("params and grad shapes must match");
    if (!is_finite(params)) return fail("params contains non-finite values");
    if (!is_finite(grad)) return fail("grad contains non-finite values");
    if (!(config.epsilon > 0.0) || !std::isfinite(config.epsilon))
        return fail("epsilon must be positive and finite");
    if (!(config.learning_rate >= 0.0) || !std::isfinite(config.learning_rate))
        return fail("learning_rate must be non-negative and finite");

    const std::size_t n = params.rows;

    WhitenResult result;
    result.ok = true;
    result.stability_epsilon = config.epsilon;
    result.normalized_grad = rms_normalize(grad);

    std::vector<double> g = gram(result.normalized_grad);
    for (std::size_t i = 0; i < n; ++i) g[i * n + i] += config.epsilon;

    std::vector<double> vectors;
    const std::vector<double> eigenvalues = jacobi_eigh(g, n, vectors);
    const std::vector<double> w = inverse_square_root(eigenvalues, vectors, n);

    result.whitening_error = whitening_error(w, g, n);
    result.whitened_direction = matmul(w, n, result.normalized_grad);

    if (config.restore_norm) {
        const double g_norm = frobenius_norm(result.normalized_grad);
        const double w_norm = frobenius_norm(result.whitened_direction);
        if (w_norm > 0.0) {
            const double scale = g_norm / w_norm;
            for (double& v : result.whitened_direction.data) v *= scale;
        }
    }

    result.updated_params = params;
    for (std::size_t i = 0; i < result.updated_params.data.size(); ++i)
        result.updated_params.data[i] -=
            config.learning_rate * result.whitened_direction.data[i];

    return result;
}

}  // namespace sw
