// Cyclic Jacobi eigendecomposition for small symmetric matrices (m <= 32).
// Returns eigenvalues in unspecified order and an orthogonal eigenvector
// matrix V whose columns are the corresponding eigenvectors, so that
// A = V * diag(lambda) * V^T up to machine precision.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "smw/matrix.hpp"

namespace smw::detail {

struct SymEig {
    std::vector<double> values;
    Matrix vectors;
    int sweeps = 0;
};

inline SymEig jacobi_eigh(Matrix a, double tol = 1e-14, int max_sweeps = 100) {
    const std::size_t m = a.rows();
    Matrix v = Matrix::zeros(m, m);
    for (std::size_t i = 0; i < m; ++i) {
        v(i, i) = 1.0;
    }

    double diagonal_scale = 0.0;
    for (std::size_t i = 0; i < m; ++i) {
        diagonal_scale = std::max(diagonal_scale, std::fabs(a(i, i)));
    }
    const double stop = tol * std::max(diagonal_scale, 1.0);

    int sweeps = 0;
    for (; sweeps < max_sweeps; ++sweeps) {
        double max_off = 0.0;
        for (std::size_t p = 0; p + 1 < m; ++p) {
            for (std::size_t q = p + 1; q < m; ++q) {
                max_off = std::max(max_off, std::fabs(a(p, q)));
            }
        }
        if (max_off <= stop) {
            break;
        }

        for (std::size_t p = 0; p + 1 < m; ++p) {
            for (std::size_t q = p + 1; q < m; ++q) {
                const double apq = a(p, q);
                if (std::fabs(apq) <= stop) {
                    a(p, q) = 0.0;
                    a(q, p) = 0.0;
                    continue;
                }

                const double app = a(p, p);
                const double aqq = a(q, q);
                const double theta = (aqq - app) / (2.0 * apq);
                const double t = (theta >= 0.0 ? 1.0 : -1.0) /
                                 (std::fabs(theta) + std::sqrt(1.0 + theta * theta));
                const double c = 1.0 / std::sqrt(1.0 + t * t);
                const double s = t * c;
                const double tau = s / (1.0 + c);

                a(p, p) = app - t * apq;
                a(q, q) = aqq + t * apq;
                a(p, q) = 0.0;
                a(q, p) = 0.0;

                for (std::size_t r = 0; r < p; ++r) {
                    const double arp = a(r, p);
                    const double arq = a(r, q);
                    a(r, p) = arp - s * (arq + tau * arp);
                    a(p, r) = a(r, p);
                    a(r, q) = arq + s * (arp - tau * arq);
                    a(q, r) = a(r, q);
                }
                for (std::size_t r = p + 1; r < q; ++r) {
                    const double apr = a(p, r);
                    const double arq = a(r, q);
                    a(p, r) = apr - s * (arq + tau * apr);
                    a(r, p) = a(p, r);
                    a(r, q) = arq + s * (apr - tau * arq);
                    a(q, r) = a(r, q);
                }
                for (std::size_t r = q + 1; r < m; ++r) {
                    const double apr = a(p, r);
                    const double aqr = a(q, r);
                    a(p, r) = apr - s * (aqr + tau * apr);
                    a(r, p) = a(p, r);
                    a(q, r) = aqr + s * (apr - tau * aqr);
                    a(r, q) = a(q, r);
                }

                for (std::size_t r = 0; r < m; ++r) {
                    const double vrp = v(r, p);
                    const double vrq = v(r, q);
                    v(r, p) = vrp - s * (vrq + tau * vrp);
                    v(r, q) = vrq + s * (vrp - tau * vrq);
                }
            }
        }
    }

    SymEig result;
    result.values.resize(m);
    for (std::size_t i = 0; i < m; ++i) {
        result.values[i] = a(i, i);
    }
    result.vectors = std::move(v);
    result.sweeps = sweeps;
    return result;
}

}  // namespace smw::detail
