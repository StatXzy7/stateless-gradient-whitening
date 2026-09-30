#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include <cstring>

#include "smw/matrix.hpp"
#include "smw/optimizer.hpp"

#include "independent_reference.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool condition, const std::string &message) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::printf("  FAIL: %s\n", message.c_str());
    }
}

bool nearly_equal(double a, double b, double tol = 1e-12) {
    const double scale = std::max({1.0, std::fabs(a), std::fabs(b)});
    return std::fabs(a - b) <= tol * scale;
}

bool matrices_close(const smw::Matrix &a, const smw::Matrix &b,
                    double tol = 1e-11) {
    if (!a.same_shape(b)) {
        return false;
    }
    for (std::size_t k = 0; k < a.size(); ++k) {
        if (!nearly_equal(a.data()[k], b.data()[k], tol)) {
            return false;
        }
    }
    return true;
}

bool matrices_bit_equal(const smw::Matrix &a, const smw::Matrix &b) {
    if (!a.same_shape(b)) {
        return false;
    }
    for (std::size_t k = 0; k < a.size(); ++k) {
        if (a.data()[k] != b.data()[k]) {
            return false;
        }
    }
    return true;
}

bool matrices_byte_equal(const smw::Matrix &a, const smw::Matrix &b) {
    if (!a.same_shape(b)) {
        return false;
    }
    return std::memcmp(a.data(), b.data(),
                       a.size() * sizeof(double)) == 0;
}

std::vector<double> flatten(const smw::Matrix &m) {
    return std::vector<double>(m.data(), m.data() + m.size());
}

void run(const std::string &name, const std::function<void()> &test) {
    std::printf("[ RUN ] %s\n", name.c_str());
    const int before = g_failures;
    test();
    if (g_failures == before) {
        std::printf("[ PASS ] %s\n", name.c_str());
    } else {
        std::printf("[ FAIL ] %s\n", name.c_str());
    }
}

}  // namespace

int main() {
    using smw::Config;
    using smw::Matrix;
    using smw::Result;

    // 1. Real update values: 1x2 case computed by hand.
    run("real_update_values", [] {
        Matrix g = {{3.0, 4.0}};
        Matrix p = {{0.10, -0.20}};
        Config cfg{0.25, 0.5, false};

        Result r = smw::step(g, p, cfg);
        check(r.accepted, "step accepted");

        const double rms = std::sqrt(25.0 / 2.0);
        check(nearly_equal(r.normalized_gradient(0, 0), 3.0 / rms),
              "normalized row entry 0");
        check(nearly_equal(r.normalized_gradient(0, 1), 4.0 / rms),
              "normalized row entry 1");

        // After RMS normalization the row has rms 1, so its squared norm is
        // n = 2 and the row Gram entry is exactly 2; A = 2 + 0.25 = 2.25.
        const double inv_root = 1.0 / std::sqrt(2.25);
        check(nearly_equal(r.whitened_direction(0, 0),
                           (3.0 / rms) * inv_root),
              "whitened entry 0");
        check(nearly_equal(r.whitened_direction(0, 1),
                           (4.0 / rms) * inv_root),
              "whitened entry 1");

        check(nearly_equal(r.updated_parameters(0, 0),
                           0.10 - 0.5 * (3.0 / rms) * inv_root),
              "updated entry 0");
        check(nearly_equal(r.updated_parameters(0, 1),
                           -0.20 - 0.5 * (4.0 / rms) * inv_root),
              "updated entry 1");
        check(r.whitening_error < 1e-12, "whitening residual tiny");
    });

    // 2. Row scaling invariance: scaling a gradient row by any non-zero
    //    factor (positive or negative) leaves the normalized gradient and
    //    hence the whitened direction unchanged.
    run("row_scaling_invariance", [] {
        Matrix g = {{1.0, -2.0, 0.5},
                    {0.0, 3.0, -1.0}};
        Matrix p = Matrix::zeros(2, 3);
        Config cfg{1e-8, 0.1, false};

        Result base = smw::step(g, p, cfg);

        Matrix g_scaled = {{7.3 * 1.0, 7.3 * -2.0, 7.3 * 0.5},
                           {2.5 * 0.0, 2.5 * 3.0, 2.5 * -1.0}};
        Result scaled = smw::step(g_scaled, p, cfg);

        check(matrices_close(base.normalized_gradient,
                             scaled.normalized_gradient, 1e-11),
              "normalized rows invariant to positive per-row scaling");
        check(matrices_close(base.whitened_direction,
                             scaled.whitened_direction, 1e-10),
              "whitened direction invariant to positive per-row scaling");
        check(matrices_close(base.updated_parameters,
                             scaled.updated_parameters, 1e-10),
              "updated parameters invariant to positive per-row scaling");

        // A negative row scale preserves the direction up to a row sign flip.
        Matrix g_flipped = {{1.0, -2.0, 0.5},
                            {0.0, -3.0, 1.0}};
        Result flipped = smw::step(g_flipped, p, cfg);
        for (std::size_t j = 0; j < 3; ++j) {
            check(nearly_equal(flipped.normalized_gradient(0, j),
                               base.normalized_gradient(0, j), 1e-12),
                  "unflipped row identical");
            check(nearly_equal(flipped.normalized_gradient(1, j),
                               -base.normalized_gradient(1, j), 1e-12),
                  "negative row scale flips normalized row");
        }
    });

    // 3. Independent whitening reference: with overall scaling off, the
    //    produced whitener must match a Newton-iteration inverse square root
    //    computed independently in the test, and W A W must equal I.
    run("independent_whitening_reference", [] {
        Matrix g = {{1.0, -2.0, 0.5, 0.0},
                    {0.0, 3.0, -1.0, 1.0},
                    {-1.5, 0.25, 2.0, -0.5}};
        Matrix p = Matrix::zeros(3, 4);
        const double reg = 1.25e-7;
        Config cfg{reg, 0.0, false};

        Result r = smw::step(g, p, cfg);
        check(r.accepted, "step accepted");

        const std::vector<double> nrm_ref =
            smw_test::reference_normalize_rows(flatten(g), 3, 4);
        const std::vector<double> a_ref =
            smw_test::reference_regularized_gram(nrm_ref, 3, 4, reg);
        const std::vector<double> w_ref =
            smw_test::reference_inverse_sqrt(a_ref, 3);

        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                check(nearly_equal(r.whitener(i, j),
                                   w_ref[smw_test::idx(3, i, j)], 1e-9),
                      "whitener matches independent Newton reference");
            }
        }

        // Reference directions D_ref = W_ref * N.
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t col = 0; col < 4; ++col) {
                double expected = 0.0;
                for (std::size_t k = 0; k < 3; ++k) {
                    expected += w_ref[smw_test::idx(3, i, k)] *
                                nrm_ref[k * 4 + col];
                }
                check(nearly_equal(r.whitened_direction(i, col),
                                   expected, 1e-9),
                      "direction matches independent reference");
            }
        }

        // W A W = I residual from the reference matrices.
        const std::vector<double> wa =
            smw_test::mat_mul(w_ref, a_ref, 3);
        const std::vector<double> waw =
            smw_test::mat_mul(wa, w_ref, 3);
        double residual = 0.0;
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                const double e =
                    waw[smw_test::idx(3, i, j)] - (i == j ? 1.0 : 0.0);
                residual = std::max(residual, std::fabs(e));
            }
        }
        check(residual < 1e-9, "reference W A W ~ I");
        check(r.whitening_error < 1e-10, "reported whitening error tiny");
    });

    // 4. Rank-deficient and near-correlated rows stay finite thanks to the
    //    positive stability parameter; the inverse square root is genuinely
    //    computed and must not silently degenerate to the identity.
    run("rank_deficient_and_near_correlated", [] {
        Matrix g = {{1.0, 2.0, -1.0},
                    {2.0, 4.0, -2.0},        // exactly proportional row
                    {1.0000001, 2.0, -1.0}}; // near-copy of row 0
        Matrix p = Matrix::zeros(3, 3);
        Config cfg{1e-10, 0.1, false};

        Result r = smw::step(g, p, cfg);
        check(r.accepted, "rank-deficient step accepted");

        for (std::size_t k = 0; k < r.whitened_direction.size(); ++k) {
            check(std::isfinite(r.whitened_direction.data()[k]),
                  "finite direction on rank-deficient input");
            check(std::isfinite(r.updated_parameters.data()[k]),
                  "finite update on rank-deficient input");
        }
        for (std::size_t k = 0; k < r.whitener.size(); ++k) {
            check(std::isfinite(r.whitener.data()[k]),
                  "finite whitener on rank-deficient input");
        }

        // A deficient direction has eigenvalue reg, so the corresponding
        // inverse-square-root factor is 1/sqrt(reg): the whitener is huge,
        // proving it was not replaced by the identity matrix.
        double wh_max = 0.0;
        for (std::size_t k = 0; k < r.whitener.size(); ++k) {
            wh_max = std::max(wh_max, std::fabs(r.whitener.data()[k]));
        }
        check(wh_max > 1e4,
              "deficient direction magnified by 1/sqrt(reg), not identity");
        check(r.whitening_error < 1e-12,
              "backward whitening error small despite rank deficiency");
        check(nearly_equal(r.used_regularization, 1e-10),
              "stability parameter reported");
    });

    // 5. Zero handling: all-zero gradient yields a zero update; a single
    //    zero row skips RMS division and remains zero in N and D.
    run("zero_gradient_and_zero_rows", [] {
        Config cfg{1e-8, 0.7, true};

        Matrix p0 = {{0.5, -0.25, 1.0},
                     {1.0, 2.0, 3.0}};
        Matrix g_zero = Matrix::zeros(2, 3);
        Result rz = smw::step(g_zero, p0, cfg);
        check(rz.accepted, "zero gradient accepted");
        check(matrices_bit_equal(rz.normalized_gradient, g_zero),
              "normalized gradient exactly zero");
        check(matrices_bit_equal(rz.whitened_direction, g_zero),
              "direction exactly zero");
        check(matrices_bit_equal(rz.updated_parameters, p0),
              "parameters unchanged for zero gradient");

        Matrix g_mix = {{0.0, 0.0, 0.0},
                        {1.0, -1.0, 0.0}};
        Result rm = smw::step(g_mix, p0, cfg);
        check(rm.accepted, "mixed zero/nonzero rows accepted");
        for (std::size_t j = 0; j < 3; ++j) {
            check(rm.normalized_gradient(0, j) == 0.0,
                  "zero row stays exactly zero after normalization");
            check(rm.whitened_direction(0, j) == 0.0,
                  "zero row stays zero in the whitened direction");
            check(rm.normalized_gradient(1, j) != 0.0 ||
                          (j == 2),
                  "nonzero row normalized");
        }
        // Only row 1 of the parameters moves.
        for (std::size_t j = 0; j < 3; ++j) {
            check(rm.updated_parameters(0, j) == p0(0, j),
                  "zero row parameters untouched");
            check(rm.updated_parameters(1, j) != p0(1, j) || j == 2,
                  "nonzero row parameters updated");
        }
    });

    // 6. No historical state: after interleaving unrelated calls, identical
    //    (gradient, config) pairs produce bit-identical directions.
    run("no_historical_state", [] {
        Config cfg{1e-8, 0.13, true};
        Matrix g = {{2.0, -1.0, 0.5},
                    {0.3, 1.2, -0.7}};
        Matrix p = {{0.1, 0.2, 0.3},
                    {-0.1, -0.2, -0.3}};

        Result first = smw::step(g, p, cfg);

        Matrix other_g = {{3.0, 0.0, -2.0},
                          {1.0, 1.0, 1.0}};
        Matrix other_p = Matrix::zeros(2, 3);
        for (int k = 0; k < 5; ++k) {
            Config noisy{1e-3 + 1e-4 * k, 0.01 * (k + 1), (k % 2) == 0};
            smw::step(other_g, other_p, noisy);
            smw::step(other_g, other_p, noisy);
        }

        Result second = smw::step(g, p, cfg);
        check(matrices_bit_equal(first.normalized_gradient,
                                 second.normalized_gradient),
              "normalized gradient reproducible after other calls");
        check(matrices_bit_equal(first.whitened_direction,
                                 second.whitened_direction),
              "direction bit-identical after other calls");
        check(matrices_bit_equal(first.whitener, second.whitener),
              "whitener bit-identical after other calls");
        check(first.whitening_error == second.whitening_error,
              "error report reproducible");
    });

    // 7. Rejected inputs: the caller's parameters matrix is never modified
    //    and no update is produced.
    run("rejected_inputs_leave_parameters_unchanged", [] {
        Matrix g_ok = {{1.0, 2.0, 3.0},
                       {0.0, 1.0, -1.0}};
        auto cfg_default = [] { return Config{1e-8, 0.1, true}; };

        auto expect_reject = [&](const Matrix &gg, Matrix pp,
                                 const Config &cc, const std::string &tag) {
            const Matrix pp_saved = pp;
            Result r = smw::step(gg, pp, cc);
            check(!r.accepted, tag + ": rejected");
            check(!r.reject_reason.empty(), tag + ": reason given");
            check(matrices_byte_equal(pp, pp_saved),
                  tag + ": caller parameters untouched (bytewise)");
        };

        expect_reject(Matrix(3, 2, 1.0), Matrix(3, 2), cfg_default(),
                      "rows greater than columns");
        expect_reject(Matrix(1, 33, 1.0), Matrix(1, 33), cfg_default(),
                      "columns exceed 32");
        expect_reject(Matrix(33, 33, 1.0), Matrix(33, 33), cfg_default(),
                      "both dimensions exceed 32");
        expect_reject(Matrix(0, 0), Matrix(0, 0), cfg_default(),
                      "empty matrices");

        Matrix p = {{0.4, 0.5, 0.6},
                    {-0.7, -0.8, -0.9}};
        Matrix g_nan = g_ok;
        g_nan(1, 1) = std::numeric_limits<double>::quiet_NaN();
        expect_reject(g_nan, p, cfg_default(), "NaN gradient");
        Matrix g_inf = g_ok;
        g_inf(0, 2) = std::numeric_limits<double>::infinity();
        expect_reject(g_inf, p, cfg_default(), "infinite gradient");
        Matrix p_nan = p;
        p_nan(0, 0) = std::numeric_limits<double>::quiet_NaN();
        expect_reject(g_ok, p_nan, cfg_default(), "NaN parameters");

        Matrix shape_g = Matrix(2, 3, 1.0);
        Matrix shape_p = Matrix(2, 2, 0.0);
        expect_reject(shape_g, shape_p, cfg_default(),
                      "gradient/parameter shape mismatch");

        expect_reject(g_ok, p, Config{0.0, 0.1, true}, "zero regularization");
        expect_reject(g_ok, p, Config{-1e-8, 0.1, true},
                      "negative regularization");
        expect_reject(g_ok, p,
                      Config{std::numeric_limits<double>::quiet_NaN(),
                             0.1, true},
                      "NaN regularization");
        expect_reject(g_ok, p, Config{1e-8, -0.01, true},
                      "negative learning rate");
        expect_reject(g_ok, p,
                      Config{1e-8,
                             std::numeric_limits<double>::infinity(), true},
                      "infinite learning rate");
    });

    // 8. Learning rate zero: the direction is still computed and inspectable
    //    while the parameters remain exactly the original values.
    run("zero_learning_rate_inspects_direction", [] {
        Matrix g = {{1.0, -2.0, 0.5},
                    {0.0, 3.0, -1.0}};
        Matrix p = {{0.1, 0.2, 0.3},
                    {0.4, 0.5, 0.6}};
        Config cfg{1e-8, 0.0, true};

        Result r = smw::step(g, p, cfg);
        check(r.accepted, "zero learning rate accepted");
        check(matrices_bit_equal(r.updated_parameters, p),
              "parameters unchanged at zero learning rate");
        bool direction_nonzero = false;
        for (std::size_t k = 0; k < r.whitened_direction.size(); ++k) {
            direction_nonzero =
                direction_nonzero || r.whitened_direction.data()[k] != 0.0;
        }
        check(direction_nonzero, "direction still computed at zero lr");
    });

    // 9. Overall-scale switch: on, the whitened direction carries the same
    //    total Frobenius norm as the normalized gradient; off, it does not
    //    (rows are decorrelated) and matches the independent reference.
    run("overall_scale_switch", [] {
        Matrix g = {{2.0, -1.0, 0.0},
                    {2.0, 0.0, -1.0}};
        Matrix p = Matrix::zeros(2, 3);

        Result on = smw::step(g, p, Config{1e-6, 0.1, true});
        Result off = smw::step(g, p, Config{1e-6, 0.1, false});

        check(nearly_equal(on.whitened_direction.frobenius(),
                           on.normalized_gradient.frobenius(), 1e-11),
              "restored direction keeps normalized gradient total norm");
        check(off.whitened_direction.frobenius() !=
                      on.whitened_direction.frobenius(),
              "unrestored direction is the raw whitened product");
        check(matrices_close(off.whitener, on.whitener, 1e-13),
              "whitener itself independent of the scale switch");
    });

    // 10. The stability term is added to the Gram matrix only: changing it
    //     never changes the normalized gradient. Also covers the maximum
    //     supported 32x32 size staying finite.
    run("regularization_only_on_gram_and_max_size", [] {
        Matrix g = Matrix::zeros(32, 32);
        for (std::size_t i = 0; i < 32; ++i) {
            for (std::size_t j = 0; j < 32; ++j) {
                g(i, j) = std::sin(0.37 * static_cast<double>(i + 1) +
                                   0.21 * static_cast<double>(j + 1));
            }
        }
        Matrix p = Matrix::zeros(32, 32);

        Result small = smw::step(g, p, Config{1e-12, 0.01, false});
        Result large = smw::step(g, p, Config{1e-2, 0.01, false});
        check(small.accepted && large.accepted, "32x32 steps accepted");
        check(matrices_bit_equal(small.normalized_gradient,
                                 large.normalized_gradient),
              "normalization independent of regularization");

        bool finite = small.whitened_direction.is_finite() &&
                      large.whitened_direction.is_finite() &&
                      small.updated_parameters.is_finite();
        check(finite, "32x32 results finite");
        check(small.whitening_error < 1e-8,
              "32x32 backward error small at tiny regularization");
        check(large.whitening_error < 1e-10,
              "32x32 backward error small at normal regularization");
    });

    // 11. A 33x33 gradient is rejected even though it is "almost" within the
    //     supported 32 limit.
    run("dimension_boundary", [] {
        Matrix g = Matrix(33, 33, 1.0);
        Matrix p = Matrix(33, 33, 0.0);
        Result r = smw::step(g, p, Config{1e-8, 0.1, true});
        check(!r.accepted, "33x33 rejected");
    });

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
