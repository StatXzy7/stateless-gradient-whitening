// Minimal dense row-major real matrix used by the stateless matrix whitener.
// Only the small set of linear-algebra primitives needed by the project is
// provided; all work happens on CPU with the C++ standard library only.
#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

namespace smw {

class Matrix {
public:
    Matrix() = default;

    Matrix(std::size_t rows, std::size_t cols, double fill = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

    Matrix(std::initializer_list<std::initializer_list<double>> values) {
        rows_ = values.size();
        cols_ = rows_ == 0 ? 0 : values.begin()->size();
        data_.reserve(rows_ * cols_);
        for (const auto &row : values) {
            if (row.size() != cols_) {
                throw std::invalid_argument("ragged matrix initializer");
            }
            data_.insert(data_.end(), row.begin(), row.end());
        }
    }

    static Matrix zeros(std::size_t rows, std::size_t cols) {
        return Matrix(rows, cols, 0.0);
    }

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    std::size_t size() const noexcept { return data_.size(); }
    bool empty() const noexcept { return data_.empty(); }

    double *data() noexcept { return data_.data(); }
    const double *data() const noexcept { return data_.data(); }

    double &operator()(std::size_t row, std::size_t col) noexcept {
        assert(row < rows_ && col < cols_);
        return data_[row * cols_ + col];
    }
    double operator()(std::size_t row, std::size_t col) const noexcept {
        assert(row < rows_ && col < cols_);
        return data_[row * cols_ + col];
    }

    double *row_begin(std::size_t row) noexcept { return data_.data() + row * cols_; }
    double *row_end(std::size_t row) noexcept { return row_begin(row) + cols_; }
    const double *row_begin(std::size_t row) const noexcept {
        return data_.data() + row * cols_;
    }
    const double *row_end(std::size_t row) const noexcept {
        return row_begin(row) + cols_;
    }

    bool same_shape(const Matrix &other) const noexcept {
        return rows_ == other.rows_ && cols_ == other.cols_;
    }

    bool is_finite() const noexcept {
        for (double v : data_) {
            if (!std::isfinite(v)) {
                return false;
            }
        }
        return true;
    }

    double frobenius_squared() const noexcept {
        // Scaled accumulation to keep the sum well conditioned.
        double scale = 0.0;
        double scaled_sum = 0.0;
        for (double v : data_) {
            const double av = std::fabs(v);
            if (av > scale) {
                scaled_sum = 1.0 + scaled_sum * (scale / av) * (scale / av);
                scale = av;
            } else if (av > 0.0) {
                const double r = av / scale;
                scaled_sum += r * r;
            }
        }
        return scale * scale * scaled_sum;
    }

    double frobenius() const noexcept {
        return std::sqrt(frobenius_squared());
    }

private:
    std::size_t rows_ = 0;
    std::size_t cols_ = 0;
    std::vector<double> data_;
};

inline Matrix add(const Matrix &a, const Matrix &b) {
    if (!a.same_shape(b)) {
        throw std::invalid_argument("shape mismatch in add");
    }
    Matrix out(a.rows(), a.cols());
    for (std::size_t i = 0; i < a.size(); ++i) {
        out.data()[i] = a.data()[i] + b.data()[i];
    }
    return out;
}

inline Matrix scale(double s, const Matrix &a) {
    Matrix out(a.rows(), a.cols());
    for (std::size_t i = 0; i < a.size(); ++i) {
        out.data()[i] = s * a.data()[i];
    }
    return out;
}

// C = A * B, where A is m x k and B is k x n.
inline Matrix multiply(const Matrix &a, const Matrix &b) {
    if (a.cols() != b.rows()) {
        throw std::invalid_argument("shape mismatch in multiply");
    }
    Matrix out(a.rows(), b.cols());
    for (std::size_t i = 0; i < a.rows(); ++i) {
        const double *ar = a.row_begin(i);
        for (std::size_t k = 0; k < a.cols(); ++k) {
            const double aik = ar[k];
            if (aik == 0.0) {
                continue;
            }
            const double *br = b.row_begin(k);
            double *orow = out.row_begin(i);
            for (std::size_t j = 0; j < b.cols(); ++j) {
                orow[j] += aik * br[j];
            }
        }
    }
    return out;
}

// G = A * A^T (symmetric m x m row Gram matrix).
inline Matrix row_gram(const Matrix &a) {
    const std::size_t m = a.rows();
    const std::size_t n = a.cols();
    Matrix g(m, m);
    for (std::size_t i = 0; i < m; ++i) {
        const double *ri = a.row_begin(i);
        for (std::size_t j = i; j < m; ++j) {
            const double *rj = a.row_begin(j);
            double dot = 0.0;
            for (std::size_t k = 0; k < n; ++k) {
                dot += ri[k] * rj[k];
            }
            g(i, j) = dot;
            g(j, i) = dot;
        }
    }
    return g;
}

}  // namespace smw
