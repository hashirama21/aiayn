#pragma once

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "aiayn/decode.hpp"
#include "aiayn/model.hpp"
#include "aiayn/optim.hpp"

namespace aiayn {

struct Example {
  std::vector<int> src, tgt;  // raw symbols; no BOS/EOS
};

enum class Task { Copy, Reverse, Sort };

inline Task parse_task(const std::string& s) {
  if (s == "copy") return Task::Copy;
  if (s == "reverse") return Task::Reverse;
  if (s == "sort") return Task::Sort;
  throw std::invalid_argument("unknown task: " + s + " (copy | reverse | sort)");
}

class TaskGenerator {
 public:
  TaskGenerator(Task task, int n_symbols, int min_len, int max_len, std::uint64_t seed)
      : task_(task), n_sym_(n_symbols), lo_(min_len), hi_(max_len), rng_(seed) {
    if (n_symbols < 2 || min_len < 1 || max_len < min_len) throw std::invalid_argument("bad task parameters");
  }
  int vocab_size() const { return kFirstToken + n_sym_; }

  Example next() {
    std::uniform_int_distribution<int> len(lo_, hi_), sym(kFirstToken, kFirstToken + n_sym_ - 1);
    Example e;
    e.src.resize(static_cast<std::size_t>(len(rng_)));
    for (int& s : e.src) s = sym(rng_);
    e.tgt = e.src;
    if (task_ == Task::Reverse) std::reverse(e.tgt.begin(), e.tgt.end());
    if (task_ == Task::Sort) std::sort(e.tgt.begin(), e.tgt.end());
    return e;
  }

 private:
  Task task_;
  int n_sym_, lo_, hi_;
  std::mt19937_64 rng_;
};

// Decoder input (BOS + target) and expected output (target + EOS).
inline void make_io(const Example& e, std::vector<int>& tgt_in, std::vector<int>& tgt_out) {
  tgt_in.assign(1, kBos);
  tgt_in.insert(tgt_in.end(), e.tgt.begin(), e.tgt.end());
  tgt_out = e.tgt;
  tgt_out.push_back(kEos);
}

// One optimizer step on a mini-batch (one tape per example, gradients accumulated). Returns the summed loss.
inline double train_step(Transformer& m, Adam& opt, const std::vector<Example>& batch, double lr,
                         std::mt19937_64& dropout_rng) {
  double n_tokens = 0;
  for (const Example& e : batch) n_tokens += static_cast<double>(e.tgt.size()) + 1;
  m.zero_grad();
  double total = 0;
  std::vector<int> tgt_in, tgt_out;
  for (const Example& e : batch) {
    make_io(e, tgt_in, tgt_out);
    Tape t;
    Var loss = m.loss(t, e.src, static_cast<int>(e.src.size()), tgt_in, tgt_out, static_cast<int>(tgt_out.size()),
                      static_cast<Real>(n_tokens), &dropout_rng);
    total += static_cast<double>(loss->val->d[0]);
    t.backward(loss);
  }
  opt.step(m.parameters(), lr);
  return total;
}

// Fraction of fresh examples reproduced exactly. beam == nullptr: greedy decoding.
inline double exact_match(const Transformer& m, TaskGenerator& gen, int n, const BeamOptions* beam = nullptr) {
  int ok = 0;
  for (int i = 0; i < n; ++i) {
    const Example e = gen.next();
    const int L = static_cast<int>(e.src.size());
    BeamOptions o = beam ? *beam : BeamOptions();
    o.max_len = L + 5;
    const DecodeResult r = beam ? beam_search(m, e.src, L, o) : greedy_decode(m, e.src, L, L + 5);
    ok += (r.finished && r.tokens == e.tgt) ? 1 : 0;
  }
  return static_cast<double>(ok) / n;
}

}  // namespace aiayn
