#include "smw/optimizer.hpp"

#include <cmath>
#include <utility>

#include "jacobi.hpp"

namespace smw {

namespace {

// RMS of a finite row of length n, computed with a scaled sum so that rows
// with large-magnitude entries cannot overflow the accumulator.
double row_rms(const double *row, std::size_t n) {
    double scale = 0.0;
    double scaled_sum = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        const double av = std::fabs(row[k]);
        if (av > scale) {
            scaled_sum = 1.0 + scaled_sum * (scale / av) * (scale / av);
            scale = av;
        } else if (av > 0.0) {
            const double r = av / scale;
            scaled_sum += r * r;
        }
    }
    if (scale == 0.0) {
        return 0.0;
    }
    return scale * std::sqrt(scaled_sum / static_cast<double>(n));
}

// Real symmetric inverse square root via Jacobi eigendecomposition.
// Because A is positive definite (Gramian plus positive regularization),
// every eigenvalue is positive and W = V diag(1/sqrt(lambda)) V^T is finite.
// The matrix is never replaced by the identity, even when the Gramian is
// rank deficient: deficient directions simply map to 1/sqrt(regularization).
struct InverseSqrt {
    Matrix root;
    std::vector<double> eig_values;
    Matrix eig_vectors;
};

InverseSqrt inverse_sqrt_symmetric(const Matrix &a) {
    detail::SymEig eig = detail::jacobi_eigh(a);
    const std::size_t m = a.rows();
    Matrix w = Matrix::zeros(m, m);
    for (std::size_t i = 0; i < m; ++i) {
        const double inv_root = 1.0 / std::sqrt(eig.values[i]);
        for (std::size_t r = 0; r < m; ++r) {
            double *wr = w.row_begin(r);
            const double vir = eig.vectors(r, i);
            for (std::size_t c = 0; c < m; ++c) {
                wr[c] += vir * inv_root * eig.vectors(c, i);
            }
        }
    }
    return InverseSqrt{std::move(w), std::move(eig.values),
                       std::move(eig.vectors)};
}

}  // namespace

bool validate_step_inputs(const Matrix &gradient,
                          const Matrix &parameters,
                          const Config &config,
                          std::string *reason) {
    auto fail = [reason](const char *message) {
        if (reason != nullptr) {
            *reason = message;
        }
        return false;
    };

    if (gradient.empty() || parameters.empty()) {
        return fail("gradient and parameters must be non-empty");
    }
    if (gradient.rows() > kMaxDim || gradient.cols() > kMaxDim) {
        return fail("matrix dimensions must not exceed 32");
    }
    if (gradient.rows() > gradient.cols()) {
        return fail("gradient rows must not exceed columns");
    }
    if (!gradient.same_shape(parameters)) {
        return fail("gradient and parameters must have the same shape");
    }
    if (!gradient.is_finite()) {
        return fail("gradient contains non-finite entries");
    }
    if (!parameters.is_finite()) {
        return fail("parameters contain non-finite entries");
    }
    if (!std::isfinite(config.regularization)) {
        return fail("regularization must be finite");
    }
    if (config.regularization <= 0.0) {
        return fail("regularization must be positive");
    }
    if (!std::isfinite(config.learning_rate)) {
        return fail("learning rate must be finite");
    }
    if (config.learning_rate < 0.0) {
        return fail("learning rate must be non-negative");
    }
    return true;
}

Result step(const Matrix &gradient,
            const Matrix &parameters,
            const Config &config) {
    Result result;
    result.used_regularization = config.regularization;

    std::string reason;
    if (!validate_step_inputs(gradient, parameters, config, &reason)) {
        result.reject_reason = std::move(reason);
        return result;
    }

    const std::size_t m = gradient.rows();
    const std::size_t n = gradient.cols();

    // 1. Per-row RMS normalization. All-zero rows skip the division and
    //    remain exactly zero.
    Matrix normalized(m, n);
    for (std::size_t i = 0; i < m; ++i) {
        const double rms = row_rms(gradient.row_begin(i), n);
        double *dst = normalized.row_begin(i);
        if (rms == 0.0) {
            for (std::size_t k = 0; k < n; ++k) {
                dst[k] = 0.0;
            }
        } else {
            const double *src = gradient.row_begin(i);
            for (std::size_t k = 0; k < n; ++k) {
                dst[k] = src[k] / rms;
            }
        }
    }

    // 2-3. Row Gramian plus the stability term on the diagonal only.
    Matrix regularized = row_gram(normalized);
    for (std::size_t i = 0; i < m; ++i) {
        regularized(i, i) += config.regularization;
    }

    // 4. Symmetric inverse square root.
    InverseSqrt isr = inverse_sqrt_symmetric(regularized);
    const Matrix &whitener = isr.root;
    const std::vector<double> &eig_values = isr.eig_values;
    const Matrix &eig_vectors = isr.eig_vectors;

    // 5. Whiten the normalized rows and optionally restore their total norm.
    Matrix direction = multiply(whitener, normalized);

    const double normalized_norm = normalized.frobenius();
    const double direction_norm = direction.frobenius();
    if (config.restore_overall_scale && normalized_norm > 0.0 &&
        direction_norm > 0.0) {
        direction = scale(normalized_norm / direction_norm, direction);
    }

    // 6. Apply the learning rate to the parameters.
    Matrix updated(parameters.rows(), parameters.cols());
    for (std::size_t i = 0; i < parameters.size(); ++i) {
        updated.data()[i] =
            parameters.data()[i] - config.learning_rate * direction.data()[i];
    }

    // Relative backward error of the computed inverse square root: compare
    // W^2 with the independently assembled true inverse A^{-1} obtained from
    // the same eigenvalue spectrum V diag(1/lambda) V^T. This metric reflects
    // numerical accuracy without condition-number amplification.
    Matrix w_squared = multiply(whitener, whitener);
    Matrix a_inverse = Matrix::zeros(m, m);
    for (std::size_t i = 0; i < m; ++i) {
        const double inv_eig = 1.0 / eig_values[i];
        for (std::size_t r = 0; r < m; ++r) {
            const double vir = eig_vectors(r, i);
            double *row = a_inverse.row_begin(r);
            for (std::size_t c = 0; c < m; ++c) {
                row[c] += vir * inv_eig * eig_vectors(c, i);
            }
        }
    }
    double num_sq = 0.0;
    double den_sq = 0.0;
    for (std::size_t k = 0; k < w_squared.size(); ++k) {
        const double e = w_squared.data()[k] - a_inverse.data()[k];
        num_sq += e * e;
        den_sq += a_inverse.data()[k] * a_inverse.data()[k];
    }
    result.whitening_error =
        std::sqrt(num_sq / std::max(den_sq, 1e-300));

    result.normalized_gradient = std::move(normalized);
    result.whitened_direction = std::move(direction);
    result.updated_parameters = std::move(updated);
    result.whitener = std::move(whitener);
    result.accepted = true;
    return result;
}

}  // namespace smw
