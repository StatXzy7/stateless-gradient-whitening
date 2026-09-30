// Reproducible small quadratic-objective demo for the stateless optimizer.
//
// Objective (m x n matrix P):
//   f(P) = 0.5 * sum_{i,j} w_ij * (P_ij - T_ij)^2,
// with analytic gradient G_ij = w_ij * (P_ij - T_ij). The target matrix T
// and weights W are generated deterministically from (m, n), so the true
// optimum P* = T and the true minimum f* = 0 are known exactly.
//
// Usage:
//   quadratic_demo [reg] [lr0] [steps] [--m N --n N] [--nan] [--no-restore]
// Invalid shapes, non-positive reg or negative lr exit with code 2 and the
// parameters are never updated (rejection is checked before any mutation).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

#include "smw/optimizer.hpp"

namespace {

double objective(const smw::Matrix &p,
                 const smw::Matrix &target,
                 const smw::Matrix &weights) {
    double f = 0.0;
    for (std::size_t i = 0; i < p.rows(); ++i) {
        for (std::size_t j = 0; j < p.cols(); ++j) {
            const double e = p(i, j) - target(i, j);
            f += weights(i, j) * e * e;
        }
    }
    return 0.5 * f;
}

smw::Matrix gradient(const smw::Matrix &p,
                     const smw::Matrix &target,
                     const smw::Matrix &weights) {
    smw::Matrix g(p.rows(), p.cols());
    for (std::size_t i = 0; i < p.rows(); ++i) {
        for (std::size_t j = 0; j < p.cols(); ++j) {
            g(i, j) = weights(i, j) * (p(i, j) - target(i, j));
        }
    }
    return g;
}

smw::Matrix make_target(std::size_t m, std::size_t n) {
    smw::Matrix t(m, n);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            t(i, j) = 0.1 * static_cast<double>((i * 7 + j * 3) % 13) - 0.6;
        }
    }
    return t;
}

smw::Matrix make_weights(std::size_t m, std::size_t n) {
    smw::Matrix w(m, n);
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            w(i, j) = 0.5 + static_cast<double>((i * 5 + j * 3) % 7) * 0.25;
        }
    }
    return w;
}

}  // namespace

int main(int argc, char **argv) {
    smw::Config config;
    std::size_t m = 2;
    std::size_t n = 3;
    int steps = 120;
    bool inject_nan = false;
    bool lr_explicit = false;
    int positional = 0;

    for (int k = 1; k < argc; ++k) {
        const std::string arg = argv[k];
        auto next_is_value = [&] {
            return k + 1 < argc && argv[k + 1][0] != '-';
        };
        if (arg == "--m") {
            if (!next_is_value()) {
                std::fprintf(stderr, "--m requires a value\n");
                return 2;
            }
            m = static_cast<std::size_t>(std::atoi(argv[++k]));
        } else if (arg == "--n") {
            if (!next_is_value()) {
                std::fprintf(stderr, "--n requires a value\n");
                return 2;
            }
            n = static_cast<std::size_t>(std::atoi(argv[++k]));
        } else if (arg == "--nan") {
            inject_nan = true;
        } else if (arg == "--no-restore") {
            config.restore_overall_scale = false;
        } else if (positional == 0) {
            config.regularization = std::atof(arg.c_str());
            ++positional;
        } else if (positional == 1) {
            config.learning_rate = std::atof(arg.c_str());
            lr_explicit = true;
            ++positional;
        } else if (positional == 2) {
            steps = std::atoi(arg.c_str());
            ++positional;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return 2;
        }
    }

    if (!lr_explicit) {
        config.learning_rate = 0.2;  // demo default
    }

    // Shape guard shared with the optimizer's own validation.
    if (m == 0 || n == 0 || m > smw::kMaxDim || n > smw::kMaxDim ||
        m > n) {
        std::fprintf(stderr,
                     "rejected: illegal shape %zux%zu "
                     "(need 1 <= m <= n <= 32)\n", m, n);
        return 2;
    }

    const smw::Matrix target = make_target(m, n);
    const smw::Matrix weights = make_weights(m, n);

    smw::Matrix params(m, n);  // start at P0 = 0; stays zero if rejected
    smw::Matrix grad = gradient(params, target, weights);
    if (inject_nan && !grad.empty()) {
        grad(0, 0) = std::numeric_limits<double>::quiet_NaN();
    }

    // Rejection is checked before parameters can change.
    std::string reason;
    if (!smw::validate_step_inputs(grad, params, config, &reason)) {
        std::fprintf(stderr, "rejected: %s\n", reason.c_str());
        std::fprintf(stderr,
                     "parameters unchanged, f(P) = %.15g (true f* = 0)\n",
                     objective(params, target, weights));
        return 2;
    }

    std::printf("stateless quadratic demo (m=%zu n=%zu steps=%d reg=%.3g "
                "lr0=%.3g restore_scale=%d)\n",
                m, n, steps, config.regularization, config.learning_rate,
                static_cast<int>(config.restore_overall_scale));
    std::printf("true optimum P* = T (deterministic), true minimum f* = 0\n");
    std::printf("%5s %22s %22s\n", "step", "f(P)", "whitening_error");

    double last_error = 0.0;
    for (int step_index = 0; step_index <= steps; ++step_index) {
        if (step_index == 0 || step_index % 10 == 0 || step_index == steps) {
            std::printf("%5d %22.15g %22.3g\n", step_index,
                        objective(params, target, weights), last_error);
        }
        if (step_index == steps) {
            break;
        }

        // Deterministic geometric schedule; the optimizer itself remains
        // stateless - the learning rate is part of this call's config.
        smw::Config call_config = config;
        call_config.learning_rate =
            config.learning_rate * std::pow(0.92, step_index);

        grad = gradient(params, target, weights);
        smw::Result r = smw::step(grad, params, call_config);
        if (!r.accepted) {
            std::fprintf(stderr, "unexpected rejection: %s\n",
                         r.reject_reason.c_str());
            return 3;
        }
        params = r.updated_parameters;
        last_error = r.whitening_error;
    }

    const double final_f = objective(params, target, weights);
    std::printf("final f(P) = %.15g, |f(P) - f*| = %.3g\n", final_f, final_f);

    // Sanity guard for the CTest run.
    if (!(final_f < 1e-8)) {
        std::fprintf(stderr, "demo failed to converge\n");
        return 1;
    }
    return 0;
}
