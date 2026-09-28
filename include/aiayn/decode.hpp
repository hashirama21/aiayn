#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "aiayn/model.hpp"

namespace aiayn {

struct DecodeResult {
  std::vector<int> tokens;   // generated tokens, without BOS and without EOS
  double logp = 0.0;         // sum of log-probabilities (including the EOS)
  double score = 0.0;        // length-normalized score used for ranking
  bool finished = false;     // ended with EOS (false: cut off at max_len)
};

struct BeamOptions {
  int beam_size = 4;
  double alpha = 0.6;
  int max_len = -1;  // < 0: source length + 50, as in the paper
};

inline double length_penalty(int len, double alpha) { return std::pow((5.0 + len) / 6.0, alpha); }

// log p(next token | prefix, source) for every vocabulary entry.
inline std::vector<double> next_log_probs(const Transformer& m, const Mat& memory, int src_len,
                                          const std::vector<int>& prefix) {
  const Mat logits = m.decode_eval(memory, src_len, prefix);
  const Real* z = logits.row(logits.r - 1);
  double mx = z[0];
  for (int k = 1; k < logits.c; ++k) mx = std::max(mx, static_cast<double>(z[k]));
  double zsum = 0;
  for (int k = 0; k < logits.c; ++k) zsum += std::exp(z[k] - mx);
  const double lse = mx + std::log(zsum);
  std::vector<double> lp(static_cast<std::size_t>(logits.c));
  for (int k = 0; k < logits.c; ++k) lp[static_cast<std::size_t>(k)] = z[k] - lse;
  return lp;
}

namespace detail_decode {
inline int effective_max_len(const Transformer& m, int src_len, int requested) {
  int n = requested > 0 ? requested : src_len + 50;
  return std::min(n, m.hyper().max_len - 1);  // BOS also takes a decoder position
}
}  // namespace detail_decode

inline DecodeResult greedy_decode(const Transformer& m, const std::vector<int>& src, int src_len, int max_len = -1) {
  const int limit = detail_decode::effective_max_len(m, src_len, max_len);
  const Mat memory = m.encode_eval(src, src_len);
  std::vector<int> prefix{kBos};
  DecodeResult r;
  for (int step = 0; step < limit; ++step) {
    const std::vector<double> lp = next_log_probs(m, memory, src_len, prefix);
    int best = kEos;
    for (int k = kEos; k < static_cast<int>(lp.size()); ++k)
      if (lp[static_cast<std::size_t>(k)] > lp[static_cast<std::size_t>(best)]) best = k;
    r.logp += lp[static_cast<std::size_t>(best)];
    if (best == kEos) {
      r.finished = true;
      break;
    }
    prefix.push_back(best);
  }
  r.tokens.assign(prefix.begin() + 1, prefix.end());
  r.score = r.logp;
  return r;
}

inline DecodeResult beam_search(const Transformer& m, const std::vector<int>& src, int src_len,
                                const BeamOptions& opt = BeamOptions()) {
  if (opt.beam_size < 1) throw std::invalid_argument("beam_size must be >= 1");
  const int limit = detail_decode::effective_max_len(m, src_len, opt.max_len);
  const Mat memory = m.encode_eval(src, src_len);
  const int V = m.hyper().tgt_vocab;

  struct Hyp {
    std::vector<int> tokens;  // includes BOS
    double logp;
  };
  std::vector<Hyp> live{{{kBos}, 0.0}}, finished;
  auto normalized = [&](const Hyp& h) {
    return h.logp / length_penalty(static_cast<int>(h.tokens.size()) - 1, opt.alpha);
  };

  for (int step = 1; step <= limit && !live.empty(); ++step) {
    struct Cand { std::size_t hyp; int tok; double logp; };
    std::vector<Cand> cands;
    for (std::size_t i = 0; i < live.size(); ++i) {
      const std::vector<double> lp = next_log_probs(m, memory, src_len, live[i].tokens);
      for (int k = kEos; k < V; ++k) cands.push_back({i, k, live[i].logp + lp[static_cast<std::size_t>(k)]});
    }
    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
      if (a.logp != b.logp) return a.logp > b.logp;
      if (a.tok != b.tok) return a.tok < b.tok;
      return a.hyp < b.hyp;
    });

    std::vector<Hyp> next;
    for (const Cand& c : cands) {
      if (static_cast<int>(next.size()) == opt.beam_size) break;
      Hyp h = live[c.hyp];
      h.tokens.push_back(c.tok);
      h.logp = c.logp;
      (c.tok == kEos ? finished : next).push_back(std::move(h));
    }
    live = std::move(next);

    // Early termination: no extension of a live hypothesis can beat the best finished one.
    if (!live.empty() && !finished.empty()) {
      double best_fin = -std::numeric_limits<double>::infinity();
      for (const Hyp& h : finished) best_fin = std::max(best_fin, normalized(h));
      double best_live = -std::numeric_limits<double>::infinity();
      for (const Hyp& h : live) best_live = std::max(best_live, h.logp);
      if (best_fin >= best_live / length_penalty(limit, opt.alpha)) break;
    }
  }

  const std::vector<Hyp>& pool = finished.empty() ? live : finished;
  if (pool.empty()) return DecodeResult{};
  std::size_t best = 0;
  for (std::size_t i = 1; i < pool.size(); ++i)
    if (normalized(pool[i]) > normalized(pool[best])) best = i;

  DecodeResult r;
  r.finished = !finished.empty();
  r.logp = pool[best].logp;
  r.score = normalized(pool[best]);
  r.tokens.assign(pool[best].tokens.begin() + 1, pool[best].tokens.end());
  if (r.finished) r.tokens.pop_back();
  return r;
}

}  // namespace aiayn
