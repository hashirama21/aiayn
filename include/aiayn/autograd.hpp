#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#ifndef AIAYN_REAL
#define AIAYN_REAL float
#endif

namespace aiayn {

using Real = AIAYN_REAL;

struct Mat {
  int r = 0, c = 0;
  std::vector<Real> d;

  Mat() = default;
  Mat(int rows, int cols, Real fill = Real(0)) : r(rows), c(cols), d(static_cast<std::size_t>(rows) * cols, fill) {
    if (rows < 0 || cols < 0) throw std::invalid_argument("Mat: negative dimension");
  }
  Real& operator()(int i, int j) { return d[static_cast<std::size_t>(i) * c + j]; }
  const Real& operator()(int i, int j) const { return d[static_cast<std::size_t>(i) * c + j]; }
  Real* row(int i) { return d.data() + static_cast<std::size_t>(i) * c; }
  const Real* row(int i) const { return d.data() + static_cast<std::size_t>(i) * c; }
  std::size_t size() const { return d.size(); }
  void zero() { std::fill(d.begin(), d.end(), Real(0)); }
};

namespace detail {
inline void check(bool ok, const char* msg) {
  if (!ok) throw std::invalid_argument(msg);
}
}  // namespace detail

// C[r x n] (+)= A[r x k] * B[k x n]
inline void gemm_nn(Mat& C, const Mat& A, const Mat& B, bool accumulate) {
  detail::check(A.c == B.r && C.r == A.r && C.c == B.c, "gemm_nn: shape mismatch");
  if (!accumulate) C.zero();
  for (int i = 0; i < A.r; ++i) {
    Real* c = C.row(i);
    for (int k = 0; k < A.c; ++k) {
      const Real a = A(i, k);
      const Real* b = B.row(k);
      for (int j = 0; j < B.c; ++j) c[j] += a * b[j];
    }
  }
}

// C[r x n] (+)= A[r x k] * B[n x k]^T
inline void gemm_nt(Mat& C, const Mat& A, const Mat& B, bool accumulate) {
  detail::check(A.c == B.c && C.r == A.r && C.c == B.r, "gemm_nt: shape mismatch");
  if (!accumulate) C.zero();
  for (int i = 0; i < A.r; ++i) {
    const Real* a = A.row(i);
    for (int j = 0; j < B.r; ++j) {
      const Real* b = B.row(j);
      Real s = 0;
      for (int k = 0; k < A.c; ++k) s += a[k] * b[k];
      C(i, j) += s;
    }
  }
}

// C[k x n] (+)= A[r x k]^T * B[r x n]
inline void gemm_tn(Mat& C, const Mat& A, const Mat& B, bool accumulate) {
  detail::check(A.r == B.r && C.r == A.c && C.c == B.c, "gemm_tn: shape mismatch");
  if (!accumulate) C.zero();
  for (int i = 0; i < A.r; ++i) {
    const Real* b = B.row(i);
    for (int k = 0; k < A.c; ++k) {
      const Real a = A(i, k);
      Real* c = C.row(k);
      for (int j = 0; j < B.c; ++j) c[j] += a * b[j];
    }
  }
}

struct Node {
  Mat own_val, own_grad;
  Mat* val = nullptr;
  Mat* grad = nullptr;
  std::function<void()> back;
};
using Var = Node*;

class Tape {
 public:
  explicit Tape(bool grad_enabled = true) : grad_enabled_(grad_enabled) {}
  Tape(const Tape&) = delete;
  Tape& operator=(const Tape&) = delete;

  bool grad_enabled() const { return grad_enabled_; }

  Var param(Mat* value, Mat* grad) {
    nodes_.emplace_back();
    Node& n = nodes_.back();
    n.val = value;
    n.grad = grad_enabled_ ? grad : nullptr;
    return &n;
  }

  Var constant(Mat value) {
    nodes_.emplace_back();
    Node& n = nodes_.back();
    n.own_val = std::move(value);
    n.val = &n.own_val;
    return &n;
  }

  // Leaf referencing external, read-only storage that outlives the tape (no copy, no gradient).
  Var constant_ref(const Mat* value) {
    nodes_.emplace_back();
    Node& n = nodes_.back();
    n.val = const_cast<Mat*>(value);
    return &n;
  }

  Var make(int r, int c, bool needs_grad) {
    nodes_.emplace_back();
    Node& n = nodes_.back();
    n.own_val = Mat(r, c);
    n.val = &n.own_val;
    if (grad_enabled_ && needs_grad) {
      n.own_grad = Mat(r, c);
      n.grad = &n.own_grad;
    }
    return &n;
  }

  bool need(Var a) const { return grad_enabled_ && a->grad != nullptr; }
  bool need(Var a, Var b) const { return need(a) || need(b); }

  void backward(Var loss) {
    detail::check(grad_enabled_ && loss->grad != nullptr, "backward: loss has no gradient (tape without grad?)");
    detail::check(loss->val->r == 1 && loss->val->c == 1, "backward: loss must be a 1x1 matrix");
    loss->grad->d[0] += Real(1);
    for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it)
      if (it->back) it->back();
  }

  std::size_t size() const { return nodes_.size(); }

 private:
  bool grad_enabled_;
  std::deque<Node> nodes_;
};

// a[r x k] * b[k x n]
inline Var matmul(Tape& t, Var a, Var b) {
  detail::check(a->val->c == b->val->r, "matmul: shape mismatch");
  Var out = t.make(a->val->r, b->val->c, t.need(a, b));
  gemm_nn(*out->val, *a->val, *b->val, false);
  if (out->grad)
    out->back = [a, b, out] {
      if (a->grad) gemm_nt(*a->grad, *out->grad, *b->val, true);
      if (b->grad) gemm_tn(*b->grad, *a->val, *out->grad, true);
    };
  return out;
}

// a[r x k] * b[n x k]^T
inline Var matmul_nt(Tape& t, Var a, Var b) {
  detail::check(a->val->c == b->val->c, "matmul_nt: shape mismatch");
  Var out = t.make(a->val->r, b->val->r, t.need(a, b));
  gemm_nt(*out->val, *a->val, *b->val, false);
  if (out->grad)
    out->back = [a, b, out] {
      if (a->grad) gemm_nn(*a->grad, *out->grad, *b->val, true);
      if (b->grad) gemm_tn(*b->grad, *out->grad, *a->val, true);
    };
  return out;
}

inline Var add(Tape& t, Var a, Var b) {
  detail::check(a->val->r == b->val->r && a->val->c == b->val->c, "add: shape mismatch");
  Var out = t.make(a->val->r, a->val->c, t.need(a, b));
  for (std::size_t i = 0; i < out->val->size(); ++i) out->val->d[i] = a->val->d[i] + b->val->d[i];
  if (out->grad)
    out->back = [a, b, out] {
      for (std::size_t i = 0; i < out->grad->size(); ++i) {
        if (a->grad) a->grad->d[i] += out->grad->d[i];
        if (b->grad) b->grad->d[i] += out->grad->d[i];
      }
    };
  return out;
}

// a[r x c] + bias[1 x c] broadcast over rows
inline Var add_bias(Tape& t, Var a, Var bias) {
  detail::check(bias->val->r == 1 && bias->val->c == a->val->c, "add_bias: bias must be 1 x cols");
  Var out = t.make(a->val->r, a->val->c, t.need(a, bias));
  for (int i = 0; i < a->val->r; ++i)
    for (int j = 0; j < a->val->c; ++j) (*out->val)(i, j) = (*a->val)(i, j) + bias->val->d[j];
  if (out->grad)
    out->back = [a, bias, out] {
      for (int i = 0; i < out->grad->r; ++i)
        for (int j = 0; j < out->grad->c; ++j) {
          const Real g = (*out->grad)(i, j);
          if (a->grad) (*a->grad)(i, j) += g;
          if (bias->grad) bias->grad->d[j] += g;
        }
    };
  return out;
}

inline Var scale(Tape& t, Var a, Real s) {
  Var out = t.make(a->val->r, a->val->c, t.need(a));
  for (std::size_t i = 0; i < out->val->size(); ++i) out->val->d[i] = a->val->d[i] * s;
  if (out->grad)
    out->back = [a, out, s] {
      for (std::size_t i = 0; i < out->grad->size(); ++i) a->grad->d[i] += out->grad->d[i] * s;
    };
  return out;
}

inline Var relu(Tape& t, Var a) {
  Var out = t.make(a->val->r, a->val->c, t.need(a));
  for (std::size_t i = 0; i < out->val->size(); ++i) out->val->d[i] = a->val->d[i] > 0 ? a->val->d[i] : Real(0);
  if (out->grad)
    out->back = [a, out] {
      for (std::size_t i = 0; i < out->grad->size(); ++i)
        if (a->val->d[i] > 0) a->grad->d[i] += out->grad->d[i];
    };
  return out;
}

// a + table[0..r): a constant per-position table (the positional encodings).
inline Var add_rows_const(Tape& t, Var a, const Mat* table) {
  detail::check(table->c == a->val->c && table->r >= a->val->r, "add_rows_const: table too small");
  Var out = t.make(a->val->r, a->val->c, t.need(a));
  for (int i = 0; i < a->val->r; ++i)
    for (int j = 0; j < a->val->c; ++j) (*out->val)(i, j) = (*a->val)(i, j) + (*table)(i, j);
  if (out->grad)
    out->back = [a, out] {
      for (std::size_t i = 0; i < out->grad->size(); ++i) a->grad->d[i] += out->grad->d[i];
    };
  return out;
}

// y = gain * (x - mean) / sqrt(var + eps) + bias, per row.
inline Var layernorm(Tape& t, Var x, Var gain, Var bias, Real eps) {
  const int R = x->val->r, C = x->val->c;
  detail::check(gain->val->r == 1 && gain->val->c == C && bias->val->r == 1 && bias->val->c == C,
                "layernorm: gain/bias must be 1 x cols");
  Var out = t.make(R, C, t.need(x) || t.need(gain) || t.need(bias));
  auto xhat = std::make_shared<std::vector<Real>>(static_cast<std::size_t>(R) * C);
  auto rstd = std::make_shared<std::vector<Real>>(static_cast<std::size_t>(R));
  for (int i = 0; i < R; ++i) {
    double mean = 0;
    for (int j = 0; j < C; ++j) mean += (*x->val)(i, j);
    mean /= C;
    double var = 0;
    for (int j = 0; j < C; ++j) {
      const double dlt = (*x->val)(i, j) - mean;
      var += dlt * dlt;
    }
    var /= C;
    const double inv = 1.0 / std::sqrt(var + static_cast<double>(eps));
    (*rstd)[i] = static_cast<Real>(inv);
    for (int j = 0; j < C; ++j) {
      const Real h = static_cast<Real>(((*x->val)(i, j) - mean) * inv);
      (*xhat)[static_cast<std::size_t>(i) * C + j] = h;
      (*out->val)(i, j) = gain->val->d[j] * h + bias->val->d[j];
    }
  }
  if (out->grad)
    out->back = [x, gain, bias, out, xhat, rstd, R, C] {
      for (int i = 0; i < R; ++i) {
        const Real* h = xhat->data() + static_cast<std::size_t>(i) * C;
        double s1 = 0, s2 = 0;
        for (int j = 0; j < C; ++j) {
          const Real dy = (*out->grad)(i, j);
          if (gain->grad) gain->grad->d[j] += dy * h[j];
          if (bias->grad) bias->grad->d[j] += dy;
          const double dxh = static_cast<double>(dy) * gain->val->d[j];
          s1 += dxh;
          s2 += dxh * h[j];
        }
        s1 /= C;
        s2 /= C;
        if (x->grad)
          for (int j = 0; j < C; ++j) {
            const double dxh = static_cast<double>((*out->grad)(i, j)) * gain->val->d[j];
            (*x->grad)(i, j) += static_cast<Real>((*rstd)[i] * (dxh - s1 - h[j] * s2));
          }
      }
    };
  return out;
}

inline Var slice_cols(Tape& t, Var a, int c0, int c1) {
  detail::check(0 <= c0 && c0 < c1 && c1 <= a->val->c, "slice_cols: bad range");
  Var out = t.make(a->val->r, c1 - c0, t.need(a));
  for (int i = 0; i < a->val->r; ++i)
    for (int j = c0; j < c1; ++j) (*out->val)(i, j - c0) = (*a->val)(i, j);
  if (out->grad)
    out->back = [a, out, c0, c1] {
      for (int i = 0; i < out->grad->r; ++i)
        for (int j = c0; j < c1; ++j) (*a->grad)(i, j) += (*out->grad)(i, j - c0);
    };
  return out;
}

inline Var concat_cols(Tape& t, const std::vector<Var>& parts) {
  detail::check(!parts.empty(), "concat_cols: no parts");
  const int R = parts[0]->val->r;
  int C = 0;
  bool need = false;
  for (Var p : parts) {
    detail::check(p->val->r == R, "concat_cols: row mismatch");
    C += p->val->c;
    need = need || t.need(p);
  }
  Var out = t.make(R, C, need);
  int off = 0;
  for (Var p : parts) {
    for (int i = 0; i < R; ++i)
      for (int j = 0; j < p->val->c; ++j) (*out->val)(i, off + j) = (*p->val)(i, j);
    off += p->val->c;
  }
  if (out->grad)
    out->back = [parts, out, R] {
      int o = 0;
      for (Var p : parts) {
        if (p->grad)
          for (int i = 0; i < R; ++i)
            for (int j = 0; j < p->val->c; ++j) (*p->grad)(i, j) += (*out->grad)(i, o + j);
        o += p->val->c;
      }
    };
  return out;
}

// out[i] = table[ids[i]]
inline Var embedding(Tape& t, Var table, const std::vector<int>& ids) {
  const int T = static_cast<int>(ids.size()), D = table->val->c;
  Var out = t.make(T, D, t.need(table));
  for (int i = 0; i < T; ++i) {
    detail::check(ids[i] >= 0 && ids[i] < table->val->r, "embedding: token id out of range");
    std::copy(table->val->row(ids[i]), table->val->row(ids[i]) + D, out->val->row(i));
  }
  if (out->grad)
    out->back = [table, out, ids, T, D] {
      for (int i = 0; i < T; ++i) {
        Real* g = table->grad->row(ids[i]);
        const Real* o = out->grad->row(i);
        for (int j = 0; j < D; ++j) g[j] += o[j];
      }
    };
  return out;
}

// Inverted dropout: kept units scaled by 1/(1-p). Identity when p <= 0 or rng == nullptr (evaluation).
inline Var dropout(Tape& t, Var a, Real p, std::mt19937_64* rng) {
  if (rng == nullptr || p <= 0) return a;
  detail::check(p < 1, "dropout: p must be < 1");
  Var out = t.make(a->val->r, a->val->c, t.need(a));
  auto keep = std::make_shared<std::vector<std::uint8_t>>(a->val->size());
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const Real s = Real(1) / (Real(1) - p);
  for (std::size_t i = 0; i < a->val->size(); ++i) {
    (*keep)[i] = u(*rng) >= static_cast<double>(p) ? 1 : 0;
    out->val->d[i] = (*keep)[i] ? a->val->d[i] * s : Real(0);
  }
  if (out->grad)
    out->back = [a, out, keep, s] {
      for (std::size_t i = 0; i < out->grad->size(); ++i)
        if ((*keep)[i]) a->grad->d[i] += out->grad->d[i] * s;
    };
  return out;
}

// Row-wise softmax. Entry (i, j) is allowed iff j < kv_len and (!causal || j <= i); others get probability 0.
inline Var masked_softmax(Tape& t, Var s, bool causal, int kv_len) {
  const int R = s->val->r, C = s->val->c;
  detail::check(kv_len >= 1 && kv_len <= C, "masked_softmax: kv_len out of range");
  Var out = t.make(R, C, t.need(s));
  for (int i = 0; i < R; ++i) {
    const int hi = causal ? std::min(kv_len, i + 1) : kv_len;
    Real mx = (*s->val)(i, 0);
    for (int j = 1; j < hi; ++j) mx = std::max(mx, (*s->val)(i, j));
    double z = 0;
    for (int j = 0; j < hi; ++j) {
      const Real e = std::exp((*s->val)(i, j) - mx);
      (*out->val)(i, j) = e;
      z += e;
    }
    for (int j = 0; j < hi; ++j) (*out->val)(i, j) = static_cast<Real>((*out->val)(i, j) / z);
  }
  if (out->grad)
    out->back = [s, out, causal, kv_len, R] {
      for (int i = 0; i < R; ++i) {
        const int hi = causal ? std::min(kv_len, i + 1) : kv_len;
        double dot = 0;
        for (int j = 0; j < hi; ++j) dot += static_cast<double>((*out->val)(i, j)) * (*out->grad)(i, j);
        for (int j = 0; j < hi; ++j)
          (*s->grad)(i, j) += static_cast<Real>((*out->val)(i, j) * ((*out->grad)(i, j) - dot));
      }
    };
  return out;
}

// Cross-entropy against a label-smoothed target for the first `valid_len` rows:
//   q = 1 - eps on the gold token, eps / (V - 1) on each other token.
inline Var cross_entropy(Tape& t, Var logits, const std::vector<int>& targets, int valid_len, Real smoothing,
                         Real norm) {
  const int T = logits->val->r, V = logits->val->c;
  detail::check(static_cast<int>(targets.size()) >= valid_len && valid_len <= T && valid_len >= 0,
                "cross_entropy: bad lengths");
  detail::check(smoothing >= 0 && smoothing < 1 && norm > 0 && V >= 2, "cross_entropy: bad smoothing/norm/vocab");
  Var out = t.make(1, 1, t.need(logits));
  auto probs = std::make_shared<std::vector<Real>>(static_cast<std::size_t>(T) * V, Real(0));
  const double eps = smoothing, off = eps / (V - 1), gold = 1.0 - eps;
  double total = 0;
  for (int i = 0; i < valid_len; ++i) {
    detail::check(targets[i] >= 0 && targets[i] < V, "cross_entropy: target out of range");
    const Real* z = logits->val->row(i);
    double mx = z[0];
    for (int k = 1; k < V; ++k) mx = std::max(mx, static_cast<double>(z[k]));
    double zsum = 0;
    for (int k = 0; k < V; ++k) zsum += std::exp(z[k] - mx);
    const double lse = mx + std::log(zsum);
    double li = 0;
    for (int k = 0; k < V; ++k) {
      const double logp = z[k] - lse;
      (*probs)[static_cast<std::size_t>(i) * V + k] = static_cast<Real>(std::exp(logp));
      li -= (k == targets[i] ? gold : off) * logp;
    }
    total += li;
  }
  out->val->d[0] = static_cast<Real>(total / norm);
  if (out->grad)
    out->back = [logits, out, probs, targets, valid_len, V, off, gold, norm] {
      const double g = static_cast<double>(out->grad->d[0]) / norm;
      for (int i = 0; i < valid_len; ++i)
        for (int k = 0; k < V; ++k) {
          const double q = (k == targets[i]) ? gold : off;
          (*logits->grad)(i, k) +=
              static_cast<Real>((static_cast<double>((*probs)[static_cast<std::size_t>(i) * V + k]) - q) * g);
        }
    };
  return out;
}

// Sum of -log p(target) over the first `valid_len` rows (for perplexity / scoring).
inline double sequence_nll(const Mat& logits, const std::vector<int>& targets, int valid_len) {
  double total = 0;
  for (int i = 0; i < valid_len; ++i) {
    const Real* z = logits.row(i);
    double mx = z[0];
    for (int k = 1; k < logits.c; ++k) mx = std::max(mx, static_cast<double>(z[k]));
    double zsum = 0;
    for (int k = 0; k < logits.c; ++k) zsum += std::exp(z[k] - mx);
    total -= (z[targets[i]] - mx) - std::log(zsum);
  }
  return total;
}

}  // namespace aiayn
