// Independent reference implementation used only by the test suite.
//
// It deliberately uses a completely different numerical route than the
// optimizer: A^{-1/2} is computed by a scaled Newton (denman-beavers style)
// iteration instead of a Jacobi eigendecomposition, and the row RMS / Gram
// calculations are written independently with plain vectors.
#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

namespace smw_test {

inline std::size_t idx(std::size_t m, std::size_t i, std::size_t j) {
    return i * m + j;
}

inline std::vector<double> mat_mul(const std::vector<double> &a,
                                   const std::vector<double> &b,
                                   std::size_t m) {
    std::vector<double> c(m * m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t k = 0; k < m; ++k) {
            const double aik = a[idx(m, i, k)];
            for (std::size_t j = 0; j < m; ++j) {
                c[idx(m, i, j)] += aik * b[idx(m, k, j)];
            }
        }
    }
    return c;
}

inline double mat_max_abs(const std::vector<double> &a) {
    double v = 0.0;
    for (double x : a) {
        v = std::max(v, std::fabs(x));
    }
    return v;
}

inline std::vector<double> mat_copy(const std::vector<double> &a) {
    return a;
}

// A^{-1/2} for symmetric positive definite A via Newton iteration on
// Y_{k+1} = 0.5 * Y_k * (3 I - B Y_k^2), B = A / alpha.
// alpha is chosen as an upper bound on the largest eigenvalue (Gershgorin),
// so eigenvalues of B lie in (0, 1] and the iteration converges reliably.
inline std::vector<double> reference_inverse_sqrt(
    const std::vector<double> &a,
    std::size_t m,
    double tol = 1e-13,
    int max_iter = 200) {
    double alpha = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        double off_sum = 0.0;
        for (std::size_t j = 0; j < m; ++j) {
            if (j != i) {
                off_sum += std::fabs(a[idx(m, i, j)]);
            }
        }
        alpha = std::max(alpha, a[idx(m, i, i)] + off_sum);
    }
    if (!(alpha > 0.0)) {
        alpha = 1.0;
    }

    std::vector<double> b(m * m);
    for (std::size_t k = 0; k < b.size(); ++k) {
        b[k] = a[k] / alpha;
    }

    // Y0 = I (valid starting point since eigenvalues of B are in (0, 1]).
    std::vector<double> y(m * m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        y[idx(m, i, i)] = 1.0;
    }

    std::vector<double> residual_prev(m * m);
    for (int it = 0; it < max_iter; ++it) {
        const std::vector<double> y2 = mat_mul(y, y, m);
        std::vector<double> t = mat_mul(b, y2, m);  // B Y^2
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < m; ++j) {
                t[idx(m, i, j)] =
                    (i == j ? 3.0 : 0.0) - t[idx(m, i, j)];
            }
        }
        std::vector<double> yt = mat_mul(y, t, m);
        for (double &x : yt) {
            x *= 0.5;
        }

        std::vector<double> residual(m * m);
        for (std::size_t k = 0; k < residual.size(); ++k) {
            residual[k] = yt[k] - y[k];
        }
        const double change = mat_max_abs(residual);
        y = yt;
        if (change < tol) {
            break;
        }
        residual_prev = std::move(residual);
    }
    (void)residual_prev;

    const double scale_back = 1.0 / std::sqrt(alpha);
    for (double &x : y) {
        x *= scale_back;
    }
    return y;
}

// Independent flat row-major representation of an m x n matrix.
inline std::vector<double> reference_normalize_rows(
    const std::vector<double> &g, std::size_t m, std::size_t n) {
    std::vector<double> nrm = g;
    for (std::size_t i = 0; i < m; ++i) {
        double sq = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            sq += g[i * n + j] * g[i * n + j];
        }
        const double rms = std::sqrt(sq / static_cast<double>(n));
        if (rms > 0.0) {
            for (std::size_t j = 0; j < n; ++j) {
                nrm[i * n + j] = g[i * n + j] / rms;
            }
        }
    }
    return nrm;
}

inline std::vector<double> reference_regularized_gram(
    const std::vector<double> &nrm,
    std::size_t m, std::size_t n, double reg) {
    std::vector<double> a(m * m, 0.0);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < m; ++j) {
            double dot = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                dot += nrm[i * n + k] * nrm[j * n + k];
            }
            a[idx(m, i, j)] = dot;
        }
        a[idx(m, i, i)] += reg;
    }
    return a;
}

}  // namespace smw_test
