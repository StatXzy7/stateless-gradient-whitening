// Hand-rolled test runner (no third-party test framework).
// Returns the number of failed checks; CTest maps nonzero to FAIL.
#include <cmath>
#include <cstring>
#include <cstdio>
#include <limits>
#include <string>

#include "stateless_whitener.h"

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool cond, const char* expr, const char* file, int line) {
    ++g_checks;
    if (!cond) {
        ++g_failures;
        std::printf("FAIL %s:%d: %s\n", file, line, expr);
    }
}

#define CHECK(cond) check((cond), #cond, __FILE__, __LINE__)

bool near(double a, double b, double tol) { return std::abs(a - b) <= tol; }

bool matrix_identical(const sw::Matrix& a, const sw::Matrix& b) {
    if (a.rows != b.rows || a.cols != b.cols || a.data.size() != b.data.size())
        return false;
    return std::memcmp(a.data.data(), b.data.data(), a.data.size() * sizeof(double)) == 0;
}

bool matrix_near(const sw::Matrix& a, const sw::Matrix& b, double tol) {
    if (a.rows != b.rows || a.cols != b.cols) return false;
    for (std::size_t k = 0; k < a.data.size(); ++k)
        if (!near(a.data[k], b.data[k], tol)) return false;
    return true;
}

double frobenius(const sw::Matrix& m) {
    double s = 0.0;
    for (double v : m.data) s += v * v;
    return std::sqrt(s);
}

sw::WhitenConfig base_config() {
    sw::WhitenConfig c;
    c.learning_rate = 0.1;
    c.epsilon = 1e-3;
    c.restore_norm = true;
    return c;
}

// ---------------------------------------------------------------- zero cases
void test_zero_gradient() {
    sw::Matrix params(2, 3, {1, 2, 3, 4, 5, 6});
    sw::Matrix grad(2, 3);  // all zeros
    const auto r = sw::whiten_step(params, grad, base_config());
    CHECK(r.ok);
    CHECK(matrix_near(r.normalized_grad, grad, 0.0));
    CHECK(matrix_near(r.whitened_direction, grad, 0.0));
    CHECK(matrix_near(r.updated_params, params, 0.0));  // zero update
}

void test_zero_row_skips_division() {
    sw::Matrix params(2, 3, {1, 1, 1, 1, 1, 1});
    sw::Matrix grad(2, 3, {0, 0, 0, 3, 4, 0});  // first row is zero
    const auto r = sw::whiten_step(params, grad, base_config());
    CHECK(r.ok);
    CHECK(r.normalized_grad.at(0, 0) == 0.0 && r.normalized_grad.at(0, 1) == 0.0 &&
          r.normalized_grad.at(0, 2) == 0.0);
    CHECK(std::isfinite(r.whitened_direction.at(0, 0)));
    CHECK(near(frobenius(r.normalized_grad), std::sqrt(3.0), 1e-12));  // RMS 1 over 3 cols
}

// ------------------------------------------------------- exact update values
void test_exact_update_values() {
    // 1x2 gradient (3,4): RMS = 5/sqrt(2), normalized = (3,4)*sqrt(2)/5.
    // Gram = 2 + eps, whitened = normalized / sqrt(2 + eps).
    sw::Matrix params(1, 2, {1.0, -1.0});
    sw::Matrix grad(1, 2, {3.0, 4.0});
    sw::WhitenConfig c = base_config();
    c.restore_norm = false;
    const auto r = sw::whiten_step(params, grad, c);
    CHECK(r.ok);

    const double rms = 5.0 / std::sqrt(2.0);
    const double nx = 3.0 / rms, ny = 4.0 / rms;
    CHECK(near(r.normalized_grad.at(0, 0), nx, 1e-12));
    CHECK(near(r.normalized_grad.at(0, 1), ny, 1e-12));
    const double inv = 1.0 / std::sqrt(2.0 + c.epsilon);
    CHECK(near(r.whitened_direction.at(0, 0), nx * inv, 1e-12));
    CHECK(near(r.whitened_direction.at(0, 1), ny * inv, 1e-12));
    CHECK(near(r.updated_params.at(0, 0), 1.0 - c.learning_rate * nx * inv, 1e-12));
    CHECK(near(r.updated_params.at(0, 1), -1.0 - c.learning_rate * ny * inv, 1e-12));
    CHECK(near(r.stability_epsilon, c.epsilon, 0.0));
}

void test_row_scale_invariance() {
    sw::Matrix params(2, 3, {0.5, -0.5, 0.25, 1.0, 2.0, 3.0});
    sw::Matrix grad(2, 3, {1, -2, 2, 0.3, 0.4, -0.5});
    sw::Matrix scaled = grad;
    for (std::size_t j = 0; j < grad.cols; ++j) {
        scaled.at(0, j) *= 7.5;
        scaled.at(1, j) *= 0.01;
    }
    sw::WhitenConfig c = base_config();
    c.restore_norm = false;
    const auto r1 = sw::whiten_step(params, grad, c);
    const auto r2 = sw::whiten_step(params, scaled, c);
    CHECK(r1.ok && r2.ok);
    CHECK(matrix_near(r1.normalized_grad, r2.normalized_grad, 1e-12));
    CHECK(matrix_near(r1.whitened_direction, r2.whitened_direction, 1e-10));
    CHECK(matrix_near(r1.updated_params, r2.updated_params, 1e-10));
}

// --------------------------------------------------------- whitening quality
void test_whitening_reference_full_rank() {
    // Well-conditioned 3x4 gradient; compare against a Newton-refined inverse
    // square root reference and verify the whitened Gram is ~identity.
    sw::Matrix params(3, 4, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    sw::Matrix grad(3, 4, {1.0, 0.2, -0.4, 0.7,   //
                           -0.3, 1.1, 0.5, 0.1,   //
                           0.6, -0.2, 0.9, -1.3});
    sw::WhitenConfig c = base_config();
    c.restore_norm = false;
    const auto r = sw::whiten_step(params, grad, c);
    CHECK(r.ok);
    CHECK(r.whitening_error < 1e-8);

    // Independent reference: Newton-Schulz iteration for the inverse square
    // root of M = G G^T + eps I (a different method than the Jacobi
    // eigendecomposition used by the optimizer).
    const std::size_t n = grad.rows;
    std::vector<double> m(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t k = 0; k < grad.cols; ++k)
                m[i * n + j] += r.normalized_grad.at(i, k) * r.normalized_grad.at(j, k);
        }
    for (std::size_t i = 0; i < n; ++i) m[i * n + i] += c.epsilon;
    double m_norm = 0.0;
    for (double v : m) m_norm += v * v;
    m_norm = std::sqrt(m_norm);
    const double alpha = 1.0 / m_norm;

    std::vector<double> y(n * n), z(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) y[i * n + j] = alpha * m[i * n + j];
        z[i * n + i] = 1.0;
    }
    auto mul = [&](const std::vector<double>& a, const std::vector<double>& b) {
        std::vector<double> out(n * n, 0.0);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                for (std::size_t k = 0; k < n; ++k) out[i * n + j] += a[i * n + k] * b[k * n + j];
        return out;
    };
    for (int it = 0; it < 200; ++it) {
        const std::vector<double> zy = mul(z, y);
        std::vector<double> t(n * n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                t[i * n + j] = 0.5 * ((i == j ? 3.0 : 0.0) - zy[i * n + j]);
        y = mul(y, t);
        z = mul(t, z);
    }
    // Z converges to (alpha * M)^{-1/2}; undo the scaling to get M^{-1/2}.
    for (double& v : z) v *= std::sqrt(alpha);

    sw::Matrix ref_dir(n, grad.cols);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < grad.cols; ++j) {
            double acc = 0.0;
            for (std::size_t k = 0; k < n; ++k)
                acc += z[i * n + k] * r.normalized_grad.at(k, j);
            ref_dir.at(i, j) = acc;
        }
    CHECK(matrix_near(r.whitened_direction, ref_dir, 1e-8));
}

void test_rank_deficient_and_correlated_rows() {
    // Row 2 = 2 * row 1 (exactly rank deficient), row 3 nearly parallel.
    sw::Matrix params(3, 4, std::vector<double>(12, 0.5));
    sw::Matrix grad(3, 4, {1, 2, 3, 4,          //
                           2, 4, 6, 8,          //
                           1.001, 2, 3, 4.001});
    const auto r = sw::whiten_step(params, grad, base_config());
    CHECK(r.ok);
    for (double v : r.whitened_direction.data) CHECK(std::isfinite(v));
    for (double v : r.updated_params.data) CHECK(std::isfinite(v));
    // Gram eigenvalues: one is exactly eps -> inv sqrt ~ 31.6, far from Inf.
    CHECK(frobenius(r.whitened_direction) < 1e3);
    // Must NOT be silently replaced by identity: direction differs from
    // the plain normalized gradient.
    CHECK(!matrix_near(r.whitened_direction, r.normalized_grad, 1e-6));
}

void test_restore_norm_toggle() {
    sw::Matrix params(2, 3, {0, 0, 0, 0, 0, 0});
    sw::Matrix grad(2, 3, {1, -2, 2, 0.3, 0.4, -0.5});
    sw::WhitenConfig c = base_config();
    c.restore_norm = true;
    const auto with = sw::whiten_step(params, grad, c);
    c.restore_norm = false;
    const auto without = sw::whiten_step(params, grad, c);
    CHECK(with.ok && without.ok);
    CHECK(near(frobenius(with.whitened_direction), frobenius(with.normalized_grad), 1e-10));
    CHECK(!near(frobenius(without.whitened_direction), frobenius(without.normalized_grad),
                1e-3));
}

// ------------------------------------------------------------- statelessness
void test_no_history_state() {
    sw::Matrix params(2, 3, {1, 1, 1, 1, 1, 1});
    sw::Matrix grad(2, 3, {0.5, -1.5, 2.0, -0.25, 0.75, 1.25});
    sw::Matrix noise(2, 3, {9, -8, 7, -6, 5, -4});
    const sw::WhitenConfig c = base_config();

    const auto first = sw::whiten_step(params, grad, c);
    // Pollute any (hypothetical) hidden state with unrelated calls.
    for (int k = 0; k < 5; ++k) (void)sw::whiten_step(params, noise, c);
    const auto second = sw::whiten_step(params, grad, c);

    CHECK(first.ok && second.ok);
    CHECK(matrix_near(first.whitened_direction, second.whitened_direction, 0.0));
    CHECK(matrix_near(first.updated_params, second.updated_params, 0.0));
}

// ------------------------------------------------------------- rejections
void test_rejections_leave_params_untouched() {
    const sw::Matrix params(2, 3, {1, 2, 3, 4, 5, 6});
    const sw::Matrix grad(2, 3, {0.1, 0.2, 0.3, 0.4, 0.5, 0.6});
    const sw::WhitenConfig good = base_config();

    auto expect_reject = [&](const sw::Matrix& p, const sw::Matrix& g,
                             const sw::WhitenConfig& c, const char* name) {
        const sw::Matrix p_before = p;
        const sw::Matrix g_before = g;
        const auto r = sw::whiten_step(p, g, c);
        CHECK(!r.ok);
        CHECK(!r.error.empty());
        CHECK(matrix_identical(p, p_before));  // inputs untouched
        CHECK(matrix_identical(g, g_before));
        (void)name;
    };

    // Illegal shapes.
    expect_reject(sw::Matrix(0, 3), sw::Matrix(0, 3), good, "zero rows");
    expect_reject(sw::Matrix(4, 3), sw::Matrix(4, 3), good, "rows > cols");
    expect_reject(sw::Matrix(3, 33), sw::Matrix(3, 33), good, "cols > 32");
    expect_reject(sw::Matrix(2, 3), sw::Matrix(3, 3), good, "shape mismatch");

    // Non-finite inputs.
    sw::Matrix bad_grad = grad;
    bad_grad.at(0, 1) = std::numeric_limits<double>::infinity();
    expect_reject(params, bad_grad, good, "inf grad");
    sw::Matrix nan_params = params;
    nan_params.at(1, 2) = std::numeric_limits<double>::quiet_NaN();
    expect_reject(nan_params, grad, good, "nan params");

    // Non-positive / non-finite epsilon.
    for (double eps : {0.0, -1e-3, -1.0, std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
        sw::WhitenConfig c = good;
        c.epsilon = eps;
        expect_reject(params, grad, c, "bad epsilon");
    }

    // Negative / non-finite learning rate.
    for (double lr : {-1e-9, -1.0, std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::quiet_NaN()}) {
        sw::WhitenConfig c = good;
        c.learning_rate = lr;
        expect_reject(params, grad, c, "bad lr");
    }
}

void test_max_boundary_shapes() {
    // Largest allowed shape 32x32 and the 1x32 edge case stay finite.
    sw::Matrix params32(32, 32), grad32(32, 32);
    for (std::size_t i = 0; i < 32; ++i)
        for (std::size_t j = 0; j < 32; ++j) {
            params32.at(i, j) = 0.01 * static_cast<double>(i * 32 + j);
            grad32.at(i, j) = std::sin(0.37 * static_cast<double>(i * 32 + j)) + 0.1;
        }
    const auto r32 = sw::whiten_step(params32, grad32, base_config());
    CHECK(r32.ok);
    CHECK(r32.whitening_error < 1e-8);
    for (double v : r32.updated_params.data) CHECK(std::isfinite(v));

    sw::Matrix params1(1, 32), grad1(1, 32);
    for (std::size_t j = 0; j < 32; ++j) grad1.at(0, j) = 0.05 * static_cast<double>(j) - 0.8;
    const auto r1 = sw::whiten_step(params1, grad1, base_config());
    CHECK(r1.ok);
    for (double v : r1.updated_params.data) CHECK(std::isfinite(v));
}

void test_zero_learning_rate() {
    const sw::Matrix params(2, 3, {1, 2, 3, 4, 5, 6});
    const sw::Matrix grad(2, 3, {0.1, -0.2, 0.3, -0.4, 0.5, -0.6});
    sw::WhitenConfig c = base_config();
    c.learning_rate = 0.0;
    const auto r = sw::whiten_step(params, grad, c);
    CHECK(r.ok);                                       // direction still computed
    CHECK(frobenius(r.whitened_direction) > 0.0);      // and is meaningful
    CHECK(matrix_near(r.updated_params, params, 0.0)); // params unchanged
}

}  // namespace

int main() {
    test_zero_gradient();
    test_zero_row_skips_division();
    test_exact_update_values();
    test_row_scale_invariance();
    test_whitening_reference_full_rank();
    test_rank_deficient_and_correlated_rows();
    test_restore_norm_toggle();
    test_no_history_state();
    test_rejections_leave_params_untouched();
    test_max_boundary_shapes();
    test_zero_learning_rate();

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
