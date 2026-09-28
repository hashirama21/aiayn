#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "aiayn/autograd.hpp"

namespace aiayn {

constexpr int kPad = 0, kBos = 1, kEos = 2, kFirstToken = 3;

struct Hyper {
  int n_layers = 6;
  int d_model = 512;
  int n_heads = 8;
  int d_ff = 2048;
  int src_vocab = 0;
  int tgt_vocab = 0;
  int max_len = 512;
  Real dropout = Real(0.1);
  Real label_smoothing = Real(0.1);
  Real ln_eps = Real(1e-6);
  bool share_embeddings = false;

  int d_k() const { return d_model / n_heads; }

  void validate() const {
    auto req = [](bool ok, const char* msg) {
      if (!ok) throw std::invalid_argument(msg);
    };
    req(n_layers > 0 && d_model > 0 && n_heads > 0 && d_ff > 0 && max_len > 0, "sizes must be positive");
    req(src_vocab >= kFirstToken && tgt_vocab >= kFirstToken, "vocabularies must include the reserved tokens");
    req(d_model % n_heads == 0, "d_model must be divisible by n_heads");
    req(d_model % 2 == 0, "d_model must be even (sinusoidal encoding uses sin/cos pairs)");
    req(dropout >= 0 && dropout < 1, "dropout must be in [0, 1)");
    req(label_smoothing >= 0 && label_smoothing < 1, "label_smoothing must be in [0, 1)");
    req(!share_embeddings || src_vocab == tgt_vocab, "share_embeddings requires src_vocab == tgt_vocab");
  }

  std::size_t n_params() const {
    const std::size_t D = d_model, F = d_ff, L = n_layers;
    const std::size_t emb = share_embeddings ? tgt_vocab * D : (static_cast<std::size_t>(src_vocab) + tgt_vocab) * D;
    const std::size_t attn = 4 * D * D;
    const std::size_t ffn = 2 * D * F + F + D;
    const std::size_t ln = 2 * D;
    return emb + L * (attn + ffn + 2 * ln) + L * (2 * attn + ffn + 3 * ln);
  }

  static Hyper base(int shared_vocab) {
    Hyper h;
    h.src_vocab = h.tgt_vocab = shared_vocab;
    h.share_embeddings = true;
    return h;
  }
  static Hyper big(int shared_vocab, Real dropout = Real(0.3)) {
    Hyper h = base(shared_vocab);
    h.d_model = 1024; h.d_ff = 4096; h.n_heads = 16; h.dropout = dropout;
    return h;
  }
};

struct Param {
  std::string name;
  Mat w, g;
  Param(std::string n, int r, int c) : name(std::move(n)), w(r, c), g(r, c) {}
};

struct AttnP { Param *wq, *wk, *wv, *wo; };
struct LnP { Param *gain, *bias; };
struct FfnP { Param *w1, *b1, *w2, *b2; };
struct EncLayer { AttnP attn; LnP ln1; FfnP ffn; LnP ln2; };
struct DecLayer { AttnP self, cross; LnP ln1, ln2; FfnP ffn; LnP ln3; };

// PE(pos, 2i) = sin(pos / 10000^(2i/d_model)), PE(pos, 2i+1) = cos(...)
inline Mat positional_encoding(int max_len, int d_model) {
  Mat pe(max_len, d_model);
  for (int pos = 0; pos < max_len; ++pos)
    for (int i = 0; i < d_model / 2; ++i) {
      const double ang = pos / std::pow(10000.0, 2.0 * i / d_model);
      pe(pos, 2 * i) = static_cast<Real>(std::sin(ang));
      pe(pos, 2 * i + 1) = static_cast<Real>(std::cos(ang));
    }
  return pe;
}

class Transformer {
 public:
  Transformer(const Hyper& h, std::uint64_t seed) : h_(h) {
    h_.validate();
    pe_ = positional_encoding(h_.max_len, h_.d_model);
    std::mt19937_64 rng(seed);
    build(rng);
  }
  Transformer(const Transformer&) = delete;
  Transformer& operator=(const Transformer&) = delete;

  const Hyper& hyper() const { return h_; }
  std::vector<Param*> parameters() {
    std::vector<Param*> v;
    for (Param& p : params_) v.push_back(&p);
    return v;
  }
  std::vector<const Param*> parameters() const {
    std::vector<const Param*> v;
    for (const Param& p : params_) v.push_back(&p);
    return v;
  }
  std::size_t n_params() const {
    std::size_t n = 0;
    for (const Param& p : params_) n += p.w.size();
    return n;
  }
  void zero_grad() {
    for (Param& p : params_) p.g.zero();
  }

  // Encoder. Only the first src_len tokens of `src` are real; padding is masked out of every attention.
  Var encode(Tape& t, const std::vector<int>& src, int src_len, std::mt19937_64* rng) const {
    check_seq(src, src_len, h_.src_vocab, "source");
    Var x = embed(t, *src_emb_, src, rng);
    for (const EncLayer& L : enc_) {
      x = sublayer(t, x, mha(t, L.attn, x, x, false, src_len), L.ln1, rng);
      x = sublayer(t, x, ffn(t, L.ffn, x), L.ln2, rng);
    }
    return x;
  }

  // Decoder. `tgt_in` is the target shifted right (starts with BOS). Returns logits [T x tgt_vocab].
  Var decode(Tape& t, const std::vector<int>& tgt_in, Var memory, int src_len, std::mt19937_64* rng) const {
    check_seq(tgt_in, static_cast<int>(tgt_in.size()), h_.tgt_vocab, "target");
    const int T = static_cast<int>(tgt_in.size());
    Var y = embed(t, *tgt_emb_, tgt_in, rng);
    for (const DecLayer& L : dec_) {
      y = sublayer(t, y, mha(t, L.self, y, y, true, T), L.ln1, rng);
      y = sublayer(t, y, mha(t, L.cross, y, memory, false, src_len), L.ln2, rng);
      y = sublayer(t, y, ffn(t, L.ffn, y), L.ln3, rng);
    }
    return matmul_nt(t, y, leaf(t, *tgt_emb_));
  }

  Var loss(Tape& t, const std::vector<int>& src, int src_len, const std::vector<int>& tgt_in,
           const std::vector<int>& tgt_out, int tgt_len, Real norm, std::mt19937_64* rng) const {
    Var memory = encode(t, src, src_len, rng);
    Var logits = decode(t, tgt_in, memory, src_len, rng);
    return cross_entropy(t, logits, tgt_out, tgt_len, h_.label_smoothing, norm);
  }

  Mat encode_eval(const std::vector<int>& src, int src_len) const {
    Tape t(false);
    return *encode(t, src, src_len, nullptr)->val;
  }
  Mat decode_eval(const Mat& memory, int src_len, const std::vector<int>& tgt_in) const {
    Tape t(false);
    Var m = t.constant_ref(&memory);
    return *decode(t, tgt_in, m, src_len, nullptr)->val;
  }

  void save(const std::string& path) const {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open for writing: " + path);
    const char magic[8] = {'A', 'I', 'A', 'Y', 'N', '0', '1', 0};
    f.write(magic, 8);
    const std::int32_t ints[9] = {h_.n_layers, h_.d_model, h_.n_heads, h_.d_ff, h_.src_vocab, h_.tgt_vocab,
                                  h_.max_len, h_.share_embeddings ? 1 : 0, static_cast<std::int32_t>(sizeof(Real))};
    f.write(reinterpret_cast<const char*>(ints), sizeof(ints));
    const double reals[3] = {h_.dropout, h_.label_smoothing, h_.ln_eps};
    f.write(reinterpret_cast<const char*>(reals), sizeof(reals));
    for (const Param& p : params_)
      f.write(reinterpret_cast<const char*>(p.w.d.data()), static_cast<std::streamsize>(p.w.size() * sizeof(Real)));
    if (!f) throw std::runtime_error("write failed: " + path);
  }

  static std::unique_ptr<Transformer> load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open: " + path);
    char magic[8];
    f.read(magic, 8);
    if (!f || std::memcmp(magic, "AIAYN01", 8) != 0) throw std::runtime_error("not an aiayn checkpoint: " + path);
    std::int32_t ints[9];
    double reals[3];
    f.read(reinterpret_cast<char*>(ints), sizeof(ints));
    f.read(reinterpret_cast<char*>(reals), sizeof(reals));
    if (!f) throw std::runtime_error("truncated checkpoint header: " + path);
    if (ints[8] != static_cast<std::int32_t>(sizeof(Real)))
      throw std::runtime_error("checkpoint was written with a different floating-point precision");
    Hyper h;
    h.n_layers = ints[0]; h.d_model = ints[1]; h.n_heads = ints[2]; h.d_ff = ints[3];
    h.src_vocab = ints[4]; h.tgt_vocab = ints[5]; h.max_len = ints[6]; h.share_embeddings = ints[7] != 0;
    h.dropout = static_cast<Real>(reals[0]); h.label_smoothing = static_cast<Real>(reals[1]);
    h.ln_eps = static_cast<Real>(reals[2]);
    auto m = std::make_unique<Transformer>(h, 0);
    for (Param& p : m->params_) {
      f.read(reinterpret_cast<char*>(p.w.d.data()), static_cast<std::streamsize>(p.w.size() * sizeof(Real)));
      if (!f) throw std::runtime_error("checkpoint is truncated: " + path);
    }
    if (f.peek() != std::ifstream::traits_type::eof()) throw std::runtime_error("trailing bytes in checkpoint: " + path);
    return m;
  }

 private:
  // Parameters are only read by forward passes; gradients accumulate into Param::g, so a const model
  // can still be differentiated.
  static Var leaf(Tape& t, const Param& p) {
    return t.param(const_cast<Mat*>(&p.w), const_cast<Mat*>(&p.g));
  }

  Var embed(Tape& t, const Param& table, const std::vector<int>& ids, std::mt19937_64* rng) const {
    Var e = scale(t, embedding(t, leaf(t, table), ids), std::sqrt(static_cast<Real>(h_.d_model)));
    return dropout(t, add_rows_const(t, e, &pe_), h_.dropout, rng);
  }

  Var sublayer(Tape& t, Var x, Var sub, const LnP& ln, std::mt19937_64* rng) const {
    return layernorm(t, add(t, x, dropout(t, sub, h_.dropout, rng)), leaf(t, *ln.gain), leaf(t, *ln.bias), h_.ln_eps);
  }

  Var ffn(Tape& t, const FfnP& f, Var x) const {
    Var hid = relu(t, add_bias(t, matmul(t, x, leaf(t, *f.w1)), leaf(t, *f.b1)));
    return add_bias(t, matmul(t, hid, leaf(t, *f.w2)), leaf(t, *f.b2));
  }

  // Multi-head attention: the per-head projections are column blocks of one [d_model x d_model] matrix each.
  Var mha(Tape& t, const AttnP& a, Var xq, Var xkv, bool causal, int kv_len) const {
    const int H = h_.n_heads, dk = h_.d_k();
    Var Q = matmul(t, xq, leaf(t, *a.wq));
    Var K = matmul(t, xkv, leaf(t, *a.wk));
    Var V = matmul(t, xkv, leaf(t, *a.wv));
    const Real inv_sqrt_dk = Real(1) / std::sqrt(static_cast<Real>(dk));
    std::vector<Var> heads;
    for (int i = 0; i < H; ++i) {
      Var q = slice_cols(t, Q, i * dk, (i + 1) * dk);
      Var k = slice_cols(t, K, i * dk, (i + 1) * dk);
      Var v = slice_cols(t, V, i * dk, (i + 1) * dk);
      Var scores = scale(t, matmul_nt(t, q, k), inv_sqrt_dk);
      Var attn = masked_softmax(t, scores, causal, kv_len);
      heads.push_back(matmul(t, attn, v));
    }
    return matmul(t, concat_cols(t, heads), leaf(t, *a.wo));
  }

  void check_seq(const std::vector<int>& s, int real_len, int vocab, const char* what) const {
    if (s.empty() || real_len < 1 || real_len > static_cast<int>(s.size()))
      throw std::invalid_argument(std::string(what) + ": bad sequence length");
    if (static_cast<int>(s.size()) > h_.max_len) throw std::out_of_range(std::string(what) + ": exceeds max_len");
    for (int id : s)
      if (id < 0 || id >= vocab) throw std::out_of_range(std::string(what) + ": token id out of range");
  }

  Param& add_param(const std::string& name, int r, int c) {
    params_.emplace_back(name, r, c);
    return params_.back();
  }
  static void xavier(Param& p, std::mt19937_64& rng) {
    const double a = std::sqrt(6.0 / (p.w.r + p.w.c));
    std::uniform_real_distribution<double> u(-a, a);
    for (Real& x : p.w.d) x = static_cast<Real>(u(rng));
  }
  AttnP make_attn(const std::string& n, std::mt19937_64& rng) {
    const int D = h_.d_model;
    AttnP a{&add_param(n + ".wq", D, D), &add_param(n + ".wk", D, D), &add_param(n + ".wv", D, D),
            &add_param(n + ".wo", D, D)};
    for (Param* p : {a.wq, a.wk, a.wv, a.wo}) xavier(*p, rng);
    return a;
  }
  LnP make_ln(const std::string& n) {
    LnP l{&add_param(n + ".gain", 1, h_.d_model), &add_param(n + ".bias", 1, h_.d_model)};
    std::fill(l.gain->w.d.begin(), l.gain->w.d.end(), Real(1));
    return l;
  }
  FfnP make_ffn(const std::string& n, std::mt19937_64& rng) {
    FfnP f{&add_param(n + ".w1", h_.d_model, h_.d_ff), &add_param(n + ".b1", 1, h_.d_ff),
           &add_param(n + ".w2", h_.d_ff, h_.d_model), &add_param(n + ".b2", 1, h_.d_model)};
    xavier(*f.w1, rng);
    xavier(*f.w2, rng);
    return f;
  }
  void build(std::mt19937_64& rng) {
    // Embeddings ~ N(0, d_model^-1/2): after the sqrt(d_model) scaling the entries have unit variance.
    std::normal_distribution<double> nd(0.0, 1.0 / std::sqrt(static_cast<double>(h_.d_model)));
    auto init_emb = [&](Param& p) { for (Real& x : p.w.d) x = static_cast<Real>(nd(rng)); };
    if (h_.share_embeddings) {
      Param& e = add_param("emb", h_.tgt_vocab, h_.d_model);
      init_emb(e);
      src_emb_ = tgt_emb_ = &e;
    } else {
      src_emb_ = &add_param("src_emb", h_.src_vocab, h_.d_model);
      tgt_emb_ = &add_param("tgt_emb", h_.tgt_vocab, h_.d_model);
      init_emb(*src_emb_);
      init_emb(*tgt_emb_);
    }
    for (int l = 0; l < h_.n_layers; ++l) {
      const std::string n = "enc" + std::to_string(l);
      EncLayer L;
      L.attn = make_attn(n + ".attn", rng); L.ln1 = make_ln(n + ".ln1");
      L.ffn = make_ffn(n + ".ffn", rng);    L.ln2 = make_ln(n + ".ln2");
      enc_.push_back(L);
    }
    for (int l = 0; l < h_.n_layers; ++l) {
      const std::string n = "dec" + std::to_string(l);
      DecLayer L;
      L.self = make_attn(n + ".self", rng);   L.ln1 = make_ln(n + ".ln1");
      L.cross = make_attn(n + ".cross", rng); L.ln2 = make_ln(n + ".ln2");
      L.ffn = make_ffn(n + ".ffn", rng);      L.ln3 = make_ln(n + ".ln3");
      dec_.push_back(L);
    }
  }

  Hyper h_;
  Mat pe_;
  std::deque<Param> params_;
  Param* src_emb_ = nullptr;
  Param* tgt_emb_ = nullptr;
  std::vector<EncLayer> enc_;
  std::vector<DecLayer> dec_;
};

}  // namespace aiayn
