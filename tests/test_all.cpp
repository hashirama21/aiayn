// Self-contained test suite. Built twice (float and double, see CMakeLists.txt).
//
// The two strongest checks:
//   * test_matches_reference : an independent, naive double-precision implementation of section 3 of the
//     paper (written from the equations, sharing no code with the library) must produce the same logits.
//   * test_gradients         : reverse-mode gradients of the whole model must match central finite
//     differences (double build only; float is too coarse for finite differences).

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "aiayn/tasks.hpp"

using namespace aiayn;
namespace fs = std::filesystem;

static constexpr bool kDouble = sizeof(Real) == 8;
static int g_checks = 0, g_fail = 0;
static bool g_skipped = false;  // set by a test that cannot run in this build

#define CHECK(cond)                                                          \
  do {                                                                       \
    ++g_checks;                                                              \
    if (!(cond)) {                                                           \
      ++g_fail;                                                              \
      std::fprintf(stderr, "  FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
    }                                                                        \
  } while (0)

#define CHECK_NEAR(a, b, tol)                                                                                   \
  do {                                                                                                          \
    ++g_checks;                                                                                                 \
    const double _a = (a), _b = (b), _t = (tol);                                                                \
    if (!(std::fabs(_a - _b) <= _t)) {                                                                          \
      ++g_fail;                                                                                                 \
      std::fprintf(stderr, "  FAIL %s:%d  |%.10g - %.10g| > %g\n", __FILE__, __LINE__, _a, _b, _t);              \
    }                                                                                                           \
  } while (0)

#define CHECK_THROWS(expr)                                                                        \
  do {                                                                                            \
    ++g_checks;                                                                                   \
    bool _threw = false;                                                                          \
    try {                                                                                         \
      (void)(expr);                                                                               \
    } catch (const std::exception&) {                                                             \
      _threw = true;                                                                              \
    }                                                                                             \
    if (!_threw) {                                                                                \
      ++g_fail;                                                                                   \
      std::fprintf(stderr, "  FAIL %s:%d  expected an exception: %s\n", __FILE__, __LINE__, #expr); \
    }                                                                                             \
  } while (0)

// tolerance helper: tight in double, loose in float
static double tol(double for_double, double for_float) { return kDouble ? for_double : for_float; }

static Hyper tiny_hyper(int layers = 2, int d_model = 16, int heads = 2, int d_ff = 24, int vocab = 11,
                        bool share = true) {
  Hyper h;
  h.n_layers = layers; h.d_model = d_model; h.n_heads = heads; h.d_ff = d_ff;
  h.src_vocab = h.tgt_vocab = vocab;
  h.max_len = 32;
  h.dropout = Real(0);
  h.share_embeddings = share;
  return h;
}

static std::vector<int> tokens(int n, int vocab, int salt) {
  std::vector<int> t(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) t[static_cast<std::size_t>(i)] = kFirstToken + (i * 5 + salt * 3 + 1) % (vocab - kFirstToken);
  return t;
}

static Mat random_mat(int r, int c, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<double> nd(0.0, 1.0);
  Mat m(r, c);
  for (Real& x : m.d) x = static_cast<Real>(nd(rng));
  return m;
}

static double max_abs_diff(const Mat& a, const Mat& b) {
  if (a.r != b.r || a.c != b.c) return 1e30;
  double m = 0;
  for (std::size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs(static_cast<double>(a.d[i]) - b.d[i]));
  return m;
}

// ===========================================================================
// Kernels and building blocks
// ===========================================================================
static void test_gemm() {
  const Mat A = random_mat(5, 7, 1), B = random_mat(7, 4, 2), Bt = random_mat(4, 7, 3), A2 = random_mat(5, 4, 4);
  Mat C(5, 4), C2(5, 4), C3(7, 4);
  gemm_nn(C, A, B, false);
  gemm_nt(C2, A, Bt, false);
  gemm_tn(C3, A, A2, false);  // A^T [7x5] * A2 [5x4]
  double w1 = 0, w2 = 0, w3 = 0;
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 4; ++j) {
      double s1 = 0, s2 = 0;
      for (int k = 0; k < 7; ++k) {
        s1 += static_cast<double>(A(i, k)) * B(k, j);
        s2 += static_cast<double>(A(i, k)) * Bt(j, k);
      }
      w1 = std::max(w1, std::fabs(s1 - C(i, j)));
      w2 = std::max(w2, std::fabs(s2 - C2(i, j)));
    }
  for (int k = 0; k < 7; ++k)
    for (int j = 0; j < 4; ++j) {
      double s = 0;
      for (int i = 0; i < 5; ++i) s += static_cast<double>(A(i, k)) * A2(i, j);
      w3 = std::max(w3, std::fabs(s - C3(k, j)));
    }
  CHECK(w1 < tol(1e-12, 1e-4) && w2 < tol(1e-12, 1e-4) && w3 < tol(1e-12, 1e-4));

  Mat acc = C;  // accumulate = true adds on top
  gemm_nn(acc, A, B, true);
  CHECK(max_abs_diff(acc, [&] { Mat d = C; for (Real& x : d.d) x *= 2; return d; }()) < tol(1e-12, 1e-4));
  CHECK_THROWS(gemm_nn(C, A, A, false));  // shape mismatch
}

static void test_positional_encoding() {
  const int D = 16;
  const Mat pe = positional_encoding(64, D);
  for (int i = 0; i < D / 2; ++i) {  // position 0: sin = 0, cos = 1
    CHECK_NEAR(pe(0, 2 * i), 0.0, 1e-12);
    CHECK_NEAR(pe(0, 2 * i + 1), 1.0, 1e-12);
  }
  CHECK_NEAR(pe(3, 0), std::sin(3.0), tol(1e-12, 1e-6));  // first pair has wavelength 2*pi
  CHECK_NEAR(pe(3, 1), std::cos(3.0), tol(1e-12, 1e-6));
  CHECK_NEAR(pe(7, 4), std::sin(7.0 / std::pow(10000.0, 4.0 / D)), tol(1e-12, 1e-6));
  CHECK_NEAR(pe(7, 5), std::cos(7.0 / std::pow(10000.0, 4.0 / D)), tol(1e-12, 1e-6));

  // Section 3.5: PE(pos + k) is a linear function (a rotation) of PE(pos), for every fixed offset k.
  const int k = 5;
  double worst = 0;
  for (int pos = 0; pos < 40; ++pos)
    for (int i = 0; i < D / 2; ++i) {
      const double w = 1.0 / std::pow(10000.0, 2.0 * i / D);
      const double s = pe(pos, 2 * i), c = pe(pos, 2 * i + 1);
      worst = std::max(worst, std::fabs(pe(pos + k, 2 * i) - (s * std::cos(k * w) + c * std::sin(k * w))));
      worst = std::max(worst, std::fabs(pe(pos + k, 2 * i + 1) - (c * std::cos(k * w) - s * std::sin(k * w))));
    }
  CHECK(worst < tol(1e-10, 1e-5));
}

static void test_layernorm() {
  Tape t(false);
  Mat xm = random_mat(3, 8, 5);
  for (int j = 0; j < 8; ++j) xm(1, j) = static_cast<Real>(100.0 + 3.0 * j);  // large offset row
  Mat gm(1, 8, Real(1)), bm(1, 8, Real(0));
  Var y = layernorm(t, t.constant(xm), t.constant(gm), t.constant(bm), Real(1e-9));
  for (int i = 0; i < 3; ++i) {
    double mean = 0, var = 0;
    for (int j = 0; j < 8; ++j) mean += (*y->val)(i, j);
    mean /= 8;
    for (int j = 0; j < 8; ++j) var += ((*y->val)(i, j) - mean) * ((*y->val)(i, j) - mean);
    var /= 8;
    CHECK_NEAR(mean, 0.0, tol(1e-10, 1e-4));
    CHECK_NEAR(var, 1.0, tol(1e-6, 1e-3));
  }
  Mat g2(1, 8), b2(1, 8);
  for (int j = 0; j < 8; ++j) { g2(0, j) = Real(2); b2(0, j) = static_cast<Real>(j); }  // gain and bias are applied
  Var y2 = layernorm(t, t.constant(xm), t.constant(g2), t.constant(b2), Real(1e-9));
  for (int j = 0; j < 8; ++j) CHECK_NEAR((*y2->val)(0, j), 2.0 * (*y->val)(0, j) + j, tol(1e-9, 1e-4));
}

static void test_masked_softmax() {
  Tape t(false);
  const Mat s = random_mat(4, 6, 6);
  Var causal = masked_softmax(t, t.constant(s), true, 6);
  for (int i = 0; i < 4; ++i) {
    double sum = 0;
    for (int j = 0; j < 6; ++j) {
      sum += (*causal->val)(i, j);
      if (j > i) CHECK((*causal->val)(i, j) == 0);  // future positions get exactly zero weight
    }
    CHECK_NEAR(sum, 1.0, tol(1e-12, 1e-5));
  }
  CHECK_NEAR((*causal->val)(0, 0), 1.0, 1e-12);  // first row can only attend to itself

  Var padded = masked_softmax(t, t.constant(s), false, 3);  // only the first 3 keys are real
  for (int i = 0; i < 4; ++i) {
    double sum = 0;
    for (int j = 0; j < 6; ++j) {
      sum += (*padded->val)(i, j);
      if (j >= 3) CHECK((*padded->val)(i, j) == 0);
    }
    CHECK_NEAR(sum, 1.0, tol(1e-12, 1e-5));
  }

  Mat huge(1, 3);  // no overflow, and shift invariance
  huge(0, 0) = 1000; huge(0, 1) = 1001; huge(0, 2) = 999;
  Var p = masked_softmax(t, t.constant(huge), false, 3);
  CHECK(std::isfinite((*p->val)(0, 0)) && std::isfinite((*p->val)(0, 1)));
  CHECK_NEAR((*p->val)(0, 1) / (*p->val)(0, 0), std::exp(1.0), tol(1e-9, 1e-3));
  CHECK_THROWS(masked_softmax(t, t.constant(s), false, 0));
  CHECK_THROWS(masked_softmax(t, t.constant(s), false, 7));
}

static void test_label_smoothing_loss() {
  Mat z(2, 4);
  const double zs[2][4] = {{0.5, -1.0, 2.0, 0.1}, {1.0, 1.0, -0.5, 0.0}};
  for (int i = 0; i < 2; ++i) for (int k = 0; k < 4; ++k) z(i, k) = static_cast<Real>(zs[i][k]);
  const std::vector<int> tg{2, 0};
  auto reference = [&](double eps, int rows) {  // straight from the definition
    double total = 0;
    for (int i = 0; i < rows; ++i) {
      double zsum = 0;
      for (int k = 0; k < 4; ++k) zsum += std::exp(zs[i][k]);
      for (int k = 0; k < 4; ++k) {
        const double q = (k == tg[static_cast<std::size_t>(i)]) ? 1 - eps : eps / 3;
        total -= q * (zs[i][k] - std::log(zsum));
      }
    }
    return total;
  };
  for (double eps : {0.0, 0.1, 0.4}) {
    Tape t(false);
    Var l = cross_entropy(t, t.constant(z), tg, 2, static_cast<Real>(eps), Real(2));
    CHECK_NEAR(l->val->d[0], reference(eps, 2) / 2, tol(1e-12, 1e-5));
  }
  Tape t(false);  // valid_len < rows ignores the trailing rows (padding)
  CHECK_NEAR(cross_entropy(t, t.constant(z), tg, 1, Real(0.1), Real(1))->val->d[0], reference(0.1, 1), tol(1e-12, 1e-5));
  CHECK_NEAR(sequence_nll(z, tg, 2), reference(0.0, 2), 1e-5);  // eps = 0 is the plain NLL
  CHECK_THROWS(cross_entropy(t, t.constant(z), std::vector<int>{2, 9}, 2, Real(0), Real(1)));  // target out of range
}

// ===========================================================================
// Independent reference implementation of section 3 (double precision, straight from the equations)
// ===========================================================================
using Vec = std::vector<double>;
using Matr = std::vector<Vec>;

struct Reference {
  const Hyper& h;
  std::map<std::string, const Param*> P;

  Reference(const Transformer& m) : h(m.hyper()) {
    for (const Param* p : m.parameters()) P[p->name] = p;
  }
  double w(const std::string& n, int i, int j) const { return P.at(n)->w(i, j); }

  Matr linear(const std::string& n, const Matr& X, int out) const {  // X W
    Matr Y(X.size(), Vec(static_cast<std::size_t>(out), 0.0));
    for (std::size_t t = 0; t < X.size(); ++t)
      for (int o = 0; o < out; ++o)
        for (std::size_t i = 0; i < X[t].size(); ++i) Y[t][static_cast<std::size_t>(o)] += X[t][i] * w(n, static_cast<int>(i), o);
    return Y;
  }
  Matr layernorm(const std::string& n, const Matr& X) const {
    Matr Y = X;
    for (auto& row : Y) {
      double mean = 0, var = 0;
      for (double v : row) mean += v;
      mean /= static_cast<double>(row.size());
      for (double v : row) var += (v - mean) * (v - mean);
      var /= static_cast<double>(row.size());
      for (std::size_t j = 0; j < row.size(); ++j)
        row[j] = w(n + ".gain", 0, static_cast<int>(j)) * (row[j] - mean) / std::sqrt(var + h.ln_eps) + w(n + ".bias", 0, static_cast<int>(j));
    }
    return Y;
  }
  Matr ffn(const std::string& n, const Matr& X) const {  // max(0, x W1 + b1) W2 + b2
    Matr H1 = linear(n + ".w1", X, h.d_ff);
    for (auto& row : H1)
      for (std::size_t j = 0; j < row.size(); ++j) row[j] = std::max(0.0, row[j] + w(n + ".b1", 0, static_cast<int>(j)));
    Matr Y = linear(n + ".w2", H1, h.d_model);
    for (auto& row : Y)
      for (std::size_t j = 0; j < row.size(); ++j) row[j] += w(n + ".b2", 0, static_cast<int>(j));
    return Y;
  }
  // MultiHead(Q,K,V) = Concat(head_1..head_h) W^O, head_i = softmax(Q W_i^Q (K W_i^K)^T / sqrt(dk)) V W_i^V
  Matr mha(const std::string& n, const Matr& xq, const Matr& xkv, bool causal, int kv_len) const {
    const int dk = h.d_k();
    Matr concat(xq.size(), Vec(static_cast<std::size_t>(h.d_model), 0.0));
    for (int head = 0; head < h.n_heads; ++head) {
      auto proj = [&](const char* which, const Matr& X) {
        Matr Y(X.size(), Vec(static_cast<std::size_t>(dk), 0.0));
        for (std::size_t t = 0; t < X.size(); ++t)
          for (int c = 0; c < dk; ++c)
            for (int j = 0; j < h.d_model; ++j) Y[t][static_cast<std::size_t>(c)] += X[t][static_cast<std::size_t>(j)] * w(n + "." + which, j, head * dk + c);
        return Y;
      };
      const Matr Q = proj("wq", xq), K = proj("wk", xkv), V = proj("wv", xkv);
      for (std::size_t t = 0; t < xq.size(); ++t) {
        Vec s;
        for (int u = 0; u < kv_len; ++u) {
          if (causal && u > static_cast<int>(t)) break;
          double d = 0;
          for (int c = 0; c < dk; ++c) d += Q[t][static_cast<std::size_t>(c)] * K[static_cast<std::size_t>(u)][static_cast<std::size_t>(c)];
          s.push_back(d / std::sqrt(static_cast<double>(dk)));
        }
        double mx = *std::max_element(s.begin(), s.end()), z = 0;
        for (double& e : s) { e = std::exp(e - mx); z += e; }
        for (std::size_t u = 0; u < s.size(); ++u)
          for (int c = 0; c < dk; ++c) concat[t][static_cast<std::size_t>(head * dk + c)] += (s[u] / z) * V[u][static_cast<std::size_t>(c)];
      }
    }
    return linear(n + ".wo", concat, h.d_model);
  }
  Matr embed(const std::string& emb, const std::vector<int>& ids) const {
    Matr X(ids.size(), Vec(static_cast<std::size_t>(h.d_model)));
    for (std::size_t t = 0; t < ids.size(); ++t)
      for (int j = 0; j < h.d_model; ++j) {
        const double ang = static_cast<double>(t) / std::pow(10000.0, 2.0 * (j / 2) / h.d_model);
        const double pe = (j % 2 == 0) ? std::sin(ang) : std::cos(ang);
        X[t][static_cast<std::size_t>(j)] = std::sqrt(static_cast<double>(h.d_model)) * w(emb, ids[t], j) + pe;
      }
    return X;
  }
  static Matr add(const Matr& a, const Matr& b) {
    Matr c = a;
    for (std::size_t i = 0; i < a.size(); ++i) for (std::size_t j = 0; j < a[i].size(); ++j) c[i][j] += b[i][j];
    return c;
  }

  Matr logits(const std::vector<int>& src, int src_len, const std::vector<int>& tgt_in) const {
    const std::string se = h.share_embeddings ? "emb" : "src_emb", te = h.share_embeddings ? "emb" : "tgt_emb";
    Matr x = embed(se, src);
    for (int l = 0; l < h.n_layers; ++l) {
      const std::string n = "enc" + std::to_string(l);
      x = layernorm(n + ".ln1", add(x, mha(n + ".attn", x, x, false, src_len)));
      x = layernorm(n + ".ln2", add(x, ffn(n + ".ffn", x)));
    }
    Matr y = embed(te, tgt_in);
    for (int l = 0; l < h.n_layers; ++l) {
      const std::string n = "dec" + std::to_string(l);
      y = layernorm(n + ".ln1", add(y, mha(n + ".self", y, y, true, static_cast<int>(y.size()))));
      y = layernorm(n + ".ln2", add(y, mha(n + ".cross", y, x, false, src_len)));
      y = layernorm(n + ".ln3", add(y, ffn(n + ".ffn", y)));
    }
    Matr out(y.size(), Vec(static_cast<std::size_t>(h.tgt_vocab), 0.0));  // pre-softmax projection = E^T
    for (std::size_t t = 0; t < y.size(); ++t)
      for (int v = 0; v < h.tgt_vocab; ++v)
        for (int j = 0; j < h.d_model; ++j) out[t][static_cast<std::size_t>(v)] += y[t][static_cast<std::size_t>(j)] * w(te, v, j);
    return out;
  }
};

static void test_matches_reference() {
  struct Case { int layers, d_model, heads, d_ff; bool share; int src_n, src_len; };
  const Case cases[] = {{1, 8, 1, 12, true, 4, 4}, {2, 16, 2, 24, true, 6, 6}, {2, 16, 4, 20, false, 7, 5}, {3, 12, 3, 16, true, 5, 3}};
  for (const Case& c : cases) {
    const Hyper h = tiny_hyper(c.layers, c.d_model, c.heads, c.d_ff, 11, c.share);
    Transformer m(h, 40 + static_cast<unsigned>(c.layers));
    const auto src = tokens(c.src_n, 11, 1);
    std::vector<int> tin{kBos};
    for (int v : tokens(5, 11, 2)) tin.push_back(v);
    const Mat memory = m.encode_eval(src, c.src_len);
    const Mat got = m.decode_eval(memory, c.src_len, tin);
    const Matr ref = Reference(m).logits(src, c.src_len, tin);
    double worst = 0;
    for (int t = 0; t < got.r; ++t)
      for (int v = 0; v < got.c; ++v) worst = std::max(worst, std::fabs(ref[static_cast<std::size_t>(t)][static_cast<std::size_t>(v)] - got(t, v)));
    CHECK(worst < tol(1e-9, 2e-3));
  }
}

// ===========================================================================
// Gradients
// ===========================================================================
static void gradient_check(const Hyper& h, Real dropout_p, int src_n, int src_len, int tgt_n, int tgt_len, unsigned seed) {
  Hyper hh = h;
  hh.dropout = dropout_p;
  Transformer m(hh, seed);
  const auto src = tokens(src_n, h.src_vocab, 1);
  std::vector<int> tin{kBos}, tout;
  for (int v : tokens(tgt_n - 1, h.tgt_vocab, 2)) tin.push_back(v);
  tout = tokens(tgt_n, h.tgt_vocab, 4);
  const Real norm = Real(tgt_len);

  auto loss_value = [&]() {  // fresh RNG each call => identical dropout masks for every evaluation
    std::mt19937_64 rng(777);
    Tape t(false);
    return static_cast<double>(m.loss(t, src, src_len, tin, tout, tgt_len, norm, dropout_p > 0 ? &rng : nullptr)->val->d[0]);
  };

  m.zero_grad();
  {
    std::mt19937_64 rng(777);
    Tape t;
    Var l = m.loss(t, src, src_len, tin, tout, tgt_len, norm, dropout_p > 0 ? &rng : nullptr);
    t.backward(l);
  }

  std::mt19937 pick(99);
  double worst = 0;
  int checked = 0, dead_params = 0;
  for (Param* p : m.parameters()) {
    double gmax = 0;
    for (Real g : p->g.d) gmax = std::max(gmax, std::fabs(static_cast<double>(g)));
    if (gmax == 0) ++dead_params;
    std::vector<std::size_t> idx;
    if (p->w.size() <= 10) {
      for (std::size_t i = 0; i < p->w.size(); ++i) idx.push_back(i);
    } else {
      // the largest analytic gradient (guaranteed informative) plus random entries
      std::size_t arg = 0;
      for (std::size_t i = 0; i < p->g.size(); ++i) if (std::fabs(p->g.d[i]) > std::fabs(p->g.d[arg])) arg = i;
      idx.push_back(arg);
      for (int k = 0; k < 9; ++k) idx.push_back(pick() % p->w.size());
    }
    for (std::size_t i : idx) {
      const Real orig = p->w.d[i];
      const double step = 1e-6;
      p->w.d[i] = static_cast<Real>(orig + step);
      const double lp = loss_value();
      p->w.d[i] = static_cast<Real>(orig - step);
      const double lm = loss_value();
      p->w.d[i] = orig;
      const double numeric = (lp - lm) / (2 * step), analytic = p->g.d[i];
      worst = std::max(worst, std::fabs(numeric - analytic) / std::max(1e-3, std::fabs(numeric) + std::fabs(analytic)));
      ++checked;
    }
  }
  CHECK(checked > 100);
  CHECK(worst < 1e-5);
  // Vacuity guard: only the parameters that cannot receive gradient by construction may be all-zero
  // (embedding rows of unused tokens are fine; whole matrices are not).
  CHECK(dead_params == 0);
}

static void test_gradients() {
  if (!kDouble) {
    g_skipped = true;  // finite differences need double precision
    return;
  }
  Hyper shared = tiny_hyper(2, 12, 3, 16, 9, true), separate = tiny_hyper(1, 8, 2, 12, 9, false);
  shared.label_smoothing = Real(0.1);
  separate.label_smoothing = Real(0.0);
  gradient_check(shared, Real(0), 5, 5, 4, 4, 1);    // tied embeddings + label smoothing
  gradient_check(separate, Real(0), 4, 4, 3, 3, 2);  // separate embeddings, plain NLL
  gradient_check(shared, Real(0), 6, 4, 5, 3, 3);    // padded source (masked) and padded target (ignored in the loss)
  gradient_check(shared, Real(0.25), 5, 5, 4, 4, 4); // dropout with fixed masks
}

// ===========================================================================
// Model properties
// ===========================================================================
static void test_decoder_causality() {
  const Hyper h = tiny_hyper();
  Transformer m(h, 5);
  const auto src = tokens(6, 11, 1);
  const Mat memory = m.encode_eval(src, 6);
  std::vector<int> a{kBos, 4, 5, 6, 7, 8}, b = a;
  b[5] = 9;  // change only the last decoder input
  const Mat la = m.decode_eval(memory, 6, a), lb = m.decode_eval(memory, 6, b);
  for (int t = 0; t < 5; ++t)
    for (int v = 0; v < la.c; ++v) CHECK(la(t, v) == lb(t, v));  // earlier positions cannot see the future
  double diff = 0;
  for (int v = 0; v < la.c; ++v) diff = std::max(diff, std::fabs(static_cast<double>(la(5, v)) - lb(5, v)));
  CHECK(diff > 1e-3);
}

static void test_source_padding() {
  const Hyper h = tiny_hyper();
  Transformer m(h, 6);
  const std::vector<int> real{4, 5, 6}, pad_a{4, 5, 6, 7, 8}, pad_b{4, 5, 6, 10, 3};  // different junk after position 3
  const std::vector<int> tin{kBos, 5, 4};
  const Mat plain = m.decode_eval(m.encode_eval(real, 3), 3, tin);
  const Mat pa = m.decode_eval(m.encode_eval(pad_a, 3), 3, tin), pb = m.decode_eval(m.encode_eval(pad_b, 3), 3, tin);
  CHECK(max_abs_diff(pa, pb) < tol(1e-12, 1e-5));     // padding content is irrelevant
  CHECK(max_abs_diff(plain, pa) < tol(1e-10, 1e-4));  // and padding is equivalent to no padding
  const Mat unmasked = m.decode_eval(m.encode_eval(pad_a, 5), 5, tin);
  CHECK(max_abs_diff(plain, unmasked) > 1e-3);        // (sanity: without the mask the tokens do matter)
}

static void test_dropout() {
  Tape t;
  const Mat x(50, 40, Real(1));
  Var in = t.constant(x);
  CHECK(dropout(t, in, Real(0.3), nullptr) == in);  // evaluation mode is the identity
  CHECK(dropout(t, in, Real(0), nullptr) == in);
  std::mt19937_64 r1(5), r2(5), r3(6);
  Var a = dropout(t, in, Real(0.3), &r1), b = dropout(t, in, Real(0.3), &r2), c = dropout(t, in, Real(0.3), &r3);
  CHECK(max_abs_diff(*a->val, *b->val) == 0.0);  // same seed, same mask
  CHECK(max_abs_diff(*a->val, *c->val) > 0.5);   // different seed, different mask
  double sum = 0;
  int kept = 0;
  for (Real v : a->val->d) {
    CHECK(v == 0 || std::fabs(v - Real(1) / Real(0.7)) < tol(1e-9, 1e-5));  // kept units are rescaled by 1/(1-p)
    sum += v;
    kept += v != 0;
  }
  CHECK_NEAR(static_cast<double>(kept) / 2000.0, 0.7, 0.04);
  CHECK_NEAR(sum / 2000.0, 1.0, 0.08);  // expectation preserved
  CHECK_THROWS(dropout(t, in, Real(1), &r1));
}

static void test_tied_output_and_param_count() {
  for (bool share : {true, false})
    for (int layers : {1, 3}) {
      const Hyper h = tiny_hyper(layers, 16, 4, 24, 13, share);
      Transformer m(h, 1);
      CHECK(m.n_params() == h.n_params());
      std::size_t emb_params = 0;
      for (const Param* p : m.parameters()) emb_params += (p->name.find("emb") != std::string::npos) ? 1 : 0;
      CHECK(emb_params == (share ? 1u : 2u));  // no separate output matrix: it is tied to the target embedding
    }
  const double base = Hyper::base(37000).n_params() / 1e6, big = Hyper::big(37000).n_params() / 1e6;
  CHECK(base > 65.0 * 0.95 && base < 65.0 * 1.05);   // table 3: 65M
  CHECK(big > 213.0 * 0.95 && big < 213.0 * 1.05);   // table 3: 213M
  CHECK(Hyper::base(37000).d_k() == 64 && Hyper::big(37000).d_k() == 64);  // d_k = d_v = 64
}

static void test_validation() {
  Hyper h = tiny_hyper();
  h.d_model = 15;  CHECK_THROWS(Transformer(h, 1));                       // not divisible by heads / odd
  h = tiny_hyper(); h.n_heads = 3; CHECK_THROWS(Transformer(h, 1));       // 16 % 3 != 0
  h = tiny_hyper(); h.src_vocab = 2; CHECK_THROWS(Transformer(h, 1));     // reserved tokens do not fit
  h = tiny_hyper(); h.tgt_vocab = 12; CHECK_THROWS(Transformer(h, 1));    // shared embeddings, different vocabularies
  h = tiny_hyper(); h.dropout = Real(1); CHECK_THROWS(Transformer(h, 1));

  Transformer m(tiny_hyper(), 1);
  CHECK_THROWS(m.encode_eval(std::vector<int>{4, 99}, 2));                 // token out of range
  CHECK_THROWS(m.encode_eval(std::vector<int>{4, 5}, 0));                  // empty real length
  CHECK_THROWS(m.encode_eval(std::vector<int>{4, 5}, 3));                  // real length beyond the sequence
  CHECK_THROWS(m.encode_eval(std::vector<int>(33, 4), 33));                // longer than max_len
  CHECK_THROWS(m.encode_eval(std::vector<int>{}, 0));
}

static void test_checkpoint() {
  const Hyper h = tiny_hyper(2, 16, 2, 24, 11, false);
  Transformer m(h, 8);
  const fs::path p = fs::temp_directory_path() / "aiayn_test.ckpt";
  m.save(p.string());
  auto r = Transformer::load(p.string());
  CHECK(r->hyper().n_layers == 2 && r->hyper().d_model == 16 && r->hyper().share_embeddings == false);
  CHECK(r->n_params() == m.n_params());
  const auto src = tokens(5, 11, 1);
  const std::vector<int> tin{kBos, 4, 5};
  CHECK(max_abs_diff(m.decode_eval(m.encode_eval(src, 5), 5, tin), r->decode_eval(r->encode_eval(src, 5), 5, tin)) == 0.0);

  fs::resize_file(p, fs::file_size(p) - 4);  // truncated
  CHECK_THROWS(Transformer::load(p.string()));
  { std::ofstream f(p, std::ios::binary); f << "this is not a checkpoint at all"; }
  CHECK_THROWS(Transformer::load(p.string()));
  fs::remove(p);
  CHECK_THROWS(Transformer::load((fs::temp_directory_path() / "aiayn_missing.ckpt").string()));
}

// ===========================================================================
// Optimizer
// ===========================================================================
static void test_adam_and_schedule() {
  // Equation (3): linear warmup, peak at warmup_steps, inverse-square-root decay.
  const int D = 512, W = 4000;
  CHECK_NEAR(noam_lr(W, D, W), std::pow(D, -0.5) * std::pow(W, -0.5), 1e-15);
  CHECK_NEAR(noam_lr(1, D, W), std::pow(D, -0.5) * std::pow(W, -1.5), 1e-15);
  CHECK_NEAR(noam_lr(2000, D, W) / noam_lr(1000, D, W), 2.0, 1e-12);        // linear in warmup
  CHECK_NEAR(noam_lr(16000, D, W) / noam_lr(4000, D, W), 0.5, 1e-12);       // 1/sqrt(step) after
  for (int s = 1; s < W; ++s) CHECK(noam_lr(s + 1, D, W) > noam_lr(s, D, W));
  for (int s = W; s < W + 200; ++s) CHECK(noam_lr(s + 1, D, W) < noam_lr(s, D, W));

  // Adam against an independent scalar implementation of Kingma & Ba, with the paper's constants.
  Param p("w", 1, 2);
  p.w(0, 0) = Real(1.0); p.w(0, 1) = Real(-2.0);
  Adam opt;
  double w0 = 1.0, w1 = -2.0, m0 = 0, m1 = 0, v0 = 0, v1 = 0;
  const double b1 = 0.9, b2 = 0.98, eps = 1e-9;
  const double grads[4][2] = {{0.5, -0.1}, {0.2, 0.3}, {-0.4, 0.05}, {0.5, -0.1}};
  for (int t = 1; t <= 4; ++t) {
    p.g(0, 0) = static_cast<Real>(grads[t - 1][0]); p.g(0, 1) = static_cast<Real>(grads[t - 1][1]);
    const double lr = 0.01 * t;
    opt.step({&p}, lr);
    const double g[2] = {grads[t - 1][0], grads[t - 1][1]};
    double* w[2] = {&w0, &w1}; double* mm[2] = {&m0, &m1}; double* vv[2] = {&v0, &v1};
    for (int k = 0; k < 2; ++k) {
      *mm[k] = b1 * *mm[k] + (1 - b1) * g[k];
      *vv[k] = b2 * *vv[k] + (1 - b2) * g[k] * g[k];
      *w[k] -= lr * (*mm[k] / (1 - std::pow(b1, t))) / (std::sqrt(*vv[k] / (1 - std::pow(b2, t))) + eps);
    }
    CHECK_NEAR(p.w(0, 0), w0, tol(1e-12, 1e-5));
    CHECK_NEAR(p.w(0, 1), w1, tol(1e-12, 1e-5));
  }
  CHECK(opt.steps() == 4);
  // First step of Adam moves each weight by ~lr regardless of the gradient scale.
  Param q("q", 1, 1);
  q.w(0, 0) = Real(0); q.g(0, 0) = Real(123.0);
  Adam o2;
  o2.step({&q}, 0.05);
  CHECK_NEAR(q.w(0, 0), -0.05, tol(1e-9, 1e-6));
}

// ===========================================================================
// Decoding
// ===========================================================================
static void test_greedy_equals_beam1() {
  Hyper h = tiny_hyper(2, 16, 2, 24, 9, true);
  h.max_len = 24;
  Transformer m(h, 21);
  for (int salt = 0; salt < 6; ++salt) {
    const auto src = tokens(3 + salt % 4, 9, salt);
    const int L = static_cast<int>(src.size());
    const DecodeResult g = greedy_decode(m, src, L, 8);
    BeamOptions o;
    o.beam_size = 1; o.alpha = 0.6; o.max_len = 8;
    const DecodeResult b = beam_search(m, src, L, o);
    CHECK(g.tokens == b.tokens);
    CHECK(g.finished == b.finished);
    CHECK_NEAR(g.logp, b.logp, 1e-9);
  }
}

static void test_beam_matches_brute_force() {
  // Many random models and several length penalties: a wide beam performs no pruning, so beam search
  // (including its early-termination rule) must return exactly the exhaustive optimum. A large alpha
  // favors long hypotheses, which is what makes a premature termination visible.
  int early_stops_possible = 0;
  for (unsigned seed = 0; seed < 12; ++seed)
    for (double alpha : {0.0, 0.6, 1.5, 2.5}) {
      Hyper h = tiny_hyper(1, 16, 2, 24, 6, true);  // 3 reserved + 3 content tokens
      h.max_len = 16;
      Transformer m(h, 100 + seed);
      const auto src = tokens(4, 6, static_cast<int>(seed));
      const int L = 4, limit = 6;
      const Mat memory = m.encode_eval(src, L);

      // Exhaustive search over every sequence (content tokens, then EOS, at most `limit` tokens incl. EOS).
      double best = -1e300, best_logp = 0;
      std::vector<int> best_seq, prefix{kBos};
      std::function<void(double)> rec = [&](double logp) {
        const std::vector<double> lp = next_log_probs(m, memory, L, prefix);
        const int len_so_far = static_cast<int>(prefix.size()) - 1;
        const double fin = logp + lp[kEos];  // finish here with EOS
        const double score = fin / length_penalty(len_so_far + 1, alpha);
        if (score > best) { best = score; best_logp = fin; best_seq.assign(prefix.begin() + 1, prefix.end()); }
        if (len_so_far + 1 < limit)
          for (int k = kFirstToken; k < h.tgt_vocab; ++k) {
            prefix.push_back(k);
            rec(logp + lp[static_cast<std::size_t>(k)]);
            prefix.pop_back();
          }
      };
      rec(0.0);
      early_stops_possible += best_seq.size() + 1 < static_cast<std::size_t>(limit) ? 1 : 0;

      BeamOptions o;
      o.beam_size = 1000; o.alpha = alpha; o.max_len = limit;  // wider than the search space
      const DecodeResult r = beam_search(m, src, L, o);
      CHECK(r.finished);
      CHECK(r.tokens == best_seq);
      CHECK_NEAR(r.logp, best_logp, 1e-9);
      CHECK_NEAR(r.score, best, 1e-9);

      o.beam_size = 4;  // a narrow beam can be worse, never better, than the exhaustive optimum
      CHECK(beam_search(m, src, L, o).score <= best + 1e-9);
    }
  CHECK(early_stops_possible > 0);  // the optimum is sometimes shorter than the limit (early stop is exercised)
  CHECK_NEAR(length_penalty(1, 0.6), 1.0, 1e-12);  // ((5+1)/6)^alpha = 1
  CHECK_NEAR(length_penalty(7, 1.0), 2.0, 1e-12);

  Hyper h = tiny_hyper(1, 16, 2, 24, 6, true);
  Transformer m(h, 1);
  BeamOptions bad;
  bad.beam_size = 0;
  CHECK_THROWS(beam_search(m, tokens(4, 6, 0), 4, bad));
}

// ===========================================================================
// Learning
// ===========================================================================
static void test_training_learns() {
  for (Task task : {Task::Copy, Task::Reverse}) {
    TaskGenerator train(task, 6, 2, 5, 11), test(task, 6, 2, 5, 12);
    Hyper h;
    h.n_layers = 2; h.d_model = 32; h.n_heads = 2; h.d_ff = 64;
    h.src_vocab = h.tgt_vocab = train.vocab_size();
    h.share_embeddings = true;
    h.max_len = 32;
    h.dropout = Real(0.0);
    Transformer m(h, 3);
    Adam opt;
    std::mt19937_64 rng(1);
    const double before = exact_match(m, test, 100);
    double first = 0, last = 0;
    const int steps = 1000;
    for (int s = 1; s <= steps; ++s) {
      std::vector<Example> b;
      for (int k = 0; k < 16; ++k) b.push_back(train.next());
      const double loss = train_step(m, opt, b, noam_lr(s, h.d_model, 200, 0.5), rng);
      if (s == 1) first = loss;
      last = loss;
    }
    CHECK(before < 0.2);
    CHECK(last < 0.5 * first);                       // the loss really goes down
    CHECK(exact_match(m, test, 200) > 0.9);          // and the task is solved on unseen inputs (greedy)
    BeamOptions bo;                                  // paper's decoding settings
    CHECK(exact_match(m, test, 100, &bo) > 0.9);     // beam 4, alpha 0.6
  }
}

static void test_training_deterministic() {
  auto run = [] {
    TaskGenerator g(Task::Copy, 5, 2, 4, 5);
    Hyper h = tiny_hyper(1, 16, 2, 24, 8, true);
    h.dropout = Real(0.1);
    Transformer m(h, 2);
    Adam opt;
    std::mt19937_64 rng(3);
    std::vector<double> losses;
    for (int s = 1; s <= 4; ++s) {
      std::vector<Example> b;
      for (int k = 0; k < 4; ++k) b.push_back(g.next());
      losses.push_back(train_step(m, opt, b, 0.001 * s, rng));
    }
    return losses;
  };
  CHECK(run() == run());  // same seeds => bit-identical training trajectory (dropout included)
}

// ===========================================================================
int main() {
  struct T { const char* name; void (*fn)(); };
  const T tests[] = {
      {"gemm nn / nt / tn vs naive loops", test_gemm},
      {"sinusoidal positional encoding (values, linear offset property)", test_positional_encoding},
      {"layer normalization", test_layernorm},
      {"masked softmax (causal, padding, overflow safety)", test_masked_softmax},
      {"label-smoothed cross-entropy vs definition", test_label_smoothing_loss},
      {"model == independent double-precision reference (section 3)", test_matches_reference},
      {"gradients == finite differences (whole model)", test_gradients},
      {"decoder causality", test_decoder_causality},
      {"source padding is masked out", test_source_padding},
      {"dropout", test_dropout},
      {"tied embeddings and parameter counts (base 65M, big 213M)", test_tied_output_and_param_count},
      {"input validation", test_validation},
      {"checkpoint round trip", test_checkpoint},
      {"Adam and the warmup schedule (eq. 3)", test_adam_and_schedule},
      {"greedy == beam search with beam 1", test_greedy_equals_beam1},
      {"beam search == brute force (12 models, alpha 0 / 0.6 / 1.5 / 2.5)", test_beam_matches_brute_force},
      {"training solves copy and reverse", test_training_learns},
      {"training is deterministic", test_training_deterministic},
  };
  std::printf("precision: %s\n", kDouble ? "double" : "float");
  for (const T& t : tests) {
    const int before = g_fail;
    g_skipped = false;
    try {
      t.fn();
    } catch (const std::exception& e) {
      ++g_fail;
      std::fprintf(stderr, "  unexpected exception: %s\n", e.what());
    }
    std::printf("[%s] %s\n", g_fail != before ? "FAIL" : (g_skipped ? "SKIP" : "PASS"), t.name);
    std::fflush(stdout);
  }
  std::printf("\n%d checks, %d failed\n", g_checks, g_fail);
  return g_fail == 0 ? 0 : 1;
}
