#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "aiayn/model.hpp"

namespace aiayn {

// Equation (3): d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5). `step` starts at 1.
inline double noam_lr(int step, int d_model, int warmup_steps, double factor = 1.0) {
  if (step < 1) step = 1;
  const double s = static_cast<double>(step);
  return factor * std::pow(static_cast<double>(d_model), -0.5) *
         std::min(std::pow(s, -0.5), s * std::pow(static_cast<double>(warmup_steps), -1.5));
}

// Adam (Kingma & Ba) with bias correction. Moment estimates are kept in double regardless of Real.
class Adam {
 public:
  explicit Adam(double beta1 = 0.9, double beta2 = 0.98, double eps = 1e-9) : b1_(beta1), b2_(beta2), eps_(eps) {}

  int steps() const { return t_; }

  void step(const std::vector<Param*>& params, double lr) {
    if (m_.empty()) {
      for (Param* p : params) {
        m_.emplace_back(p->w.size(), 0.0);
        v_.emplace_back(p->w.size(), 0.0);
      }
    }
    if (m_.size() != params.size()) throw std::invalid_argument("Adam: parameter list changed between steps");
    ++t_;
    const double c1 = 1.0 - std::pow(b1_, t_), c2 = 1.0 - std::pow(b2_, t_);
    for (std::size_t k = 0; k < params.size(); ++k) {
      Param& p = *params[k];
      for (std::size_t i = 0; i < p.w.size(); ++i) {
        const double g = p.g.d[i];
        const double m = b1_ * m_[k][i] + (1.0 - b1_) * g;
        const double v = b2_ * v_[k][i] + (1.0 - b2_) * g * g;
        m_[k][i] = m;
        v_[k][i] = v;
        p.w.d[i] = static_cast<Real>(p.w.d[i] - lr * (m / c1) / (std::sqrt(v / c2) + eps_));
      }
    }
  }

 private:
  double b1_, b2_, eps_;
  int t_ = 0;
  std::vector<std::vector<double>> m_, v_;
};

}  // namespace aiayn
