
// Trains the Transformer of Vaswani et al. (2017) on a small synthetic task and decodes with beam search.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

#include "aiayn/tasks.hpp"

using namespace aiayn;
using Clock = std::chrono::steady_clock;

static void usage() {
  std::puts(
      "usage: train_toy [options]\n"
      "  --task copy|reverse|sort   (default reverse)\n"
      "  --symbols N       alphabet size (default 12)\n"
      "  --min-len N --max-len N    sequence lengths (default 3..8)\n"
      "  --layers N --d-model N --heads N --d-ff N   (default 2 / 64 / 4 / 128)\n"
      "  --dropout P --smoothing P  (default 0.1 / 0.1, the paper's values)\n"
      "  --steps N --batch N        (default 1500 / 32)\n"
      "  --warmup N --lr-factor F   noam schedule (default 300 / 1.0)\n"
      "  --seed S --eval-every N --beam K --alpha A\n"
      "  --save PATH | --load PATH  checkpoint (with --load and --steps 0: decode only)\n"
      "  --paper-sizes     print parameter counts of the paper's base and big models and exit");
}

int main(int argc, char** argv) {
  std::string task_name = "reverse", save_path, load_path;
  int symbols = 12, min_len = 3, max_len = 8, layers = 2, d_model = 64, heads = 4, d_ff = 128;
  int steps = 1500, batch = 32, warmup = 300, eval_every = 250, beam = 4;
  double dropout = 0.1, smoothing = 0.1, lr_factor = 1.0, alpha = 0.6;
  std::uint64_t seed = 1;
  bool paper_sizes = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&]() -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", a.c_str()); std::exit(2); }
      return argv[++i];
    };
    if (a == "--task") task_name = val();
    else if (a == "--symbols") symbols = std::atoi(val());
    else if (a == "--min-len") min_len = std::atoi(val());
    else if (a == "--max-len") max_len = std::atoi(val());
    else if (a == "--layers") layers = std::atoi(val());
    else if (a == "--d-model") d_model = std::atoi(val());
    else if (a == "--heads") heads = std::atoi(val());
    else if (a == "--d-ff") d_ff = std::atoi(val());
    else if (a == "--dropout") dropout = std::atof(val());
    else if (a == "--smoothing") smoothing = std::atof(val());
    else if (a == "--steps") steps = std::atoi(val());
    else if (a == "--batch") batch = std::atoi(val());
    else if (a == "--warmup") warmup = std::atoi(val());
    else if (a == "--lr-factor") lr_factor = std::atof(val());
    else if (a == "--seed") seed = std::strtoull(val(), nullptr, 10);
    else if (a == "--eval-every") eval_every = std::atoi(val());
    else if (a == "--beam") beam = std::atoi(val());
    else if (a == "--alpha") alpha = std::atof(val());
    else if (a == "--save") save_path = val();
    else if (a == "--load") load_path = val();
    else if (a == "--paper-sizes") paper_sizes = true;
    else if (a == "--help" || a == "-h") { usage(); return 0; }
    else { std::fprintf(stderr, "unknown option: %s\n", a.c_str()); usage(); return 2; }
  }

  try {
    if (paper_sizes) {
      const Hyper base = Hyper::base(37000), big = Hyper::big(37000);
      std::printf("shared 37000-token vocabulary, this implementation's parameter count:\n");
      std::printf("  base (N=6, d_model=512, d_ff=2048, h=8):   %.1fM   (paper, table 3: 65M)\n", base.n_params() / 1e6);
      std::printf("  big  (N=6, d_model=1024, d_ff=4096, h=16): %.1fM   (paper, table 3: 213M)\n", big.n_params() / 1e6);
      return 0;
    }

    const Task task = parse_task(task_name);
    TaskGenerator train_gen(task, symbols, min_len, max_len, seed * 7919 + 1);
    TaskGenerator test_gen(task, symbols, min_len, max_len, seed * 104729 + 3);

    std::unique_ptr<Transformer> model;
    if (!load_path.empty()) {
      model = Transformer::load(load_path);
      std::printf("loaded %s (%zu parameters)\n", load_path.c_str(), model->n_params());
    } else {
      Hyper h;
      h.n_layers = layers; h.d_model = d_model; h.n_heads = heads; h.d_ff = d_ff;
      h.src_vocab = h.tgt_vocab = train_gen.vocab_size();
      h.share_embeddings = true;
      h.max_len = std::max(64, 2 * max_len + 8);
      h.dropout = static_cast<Real>(dropout);
      h.label_smoothing = static_cast<Real>(smoothing);
      model = std::make_unique<Transformer>(h, seed);
      std::printf("task=%s  N=%d d_model=%d heads=%d d_ff=%d  vocab=%d  %zu parameters\n", task_name.c_str(), layers,
                  d_model, heads, d_ff, h.src_vocab, model->n_params());
    }
    const Hyper& h = model->hyper();

    Adam opt;  // beta1 = 0.9, beta2 = 0.98, eps = 1e-9
    std::mt19937_64 drop_rng(seed + 12345);
    const auto t0 = Clock::now();
    for (int step = 1; step <= steps; ++step) {
      std::vector<Example> b;
      for (int k = 0; k < batch; ++k) b.push_back(train_gen.next());
      const double lr = noam_lr(step, h.d_model, warmup, lr_factor);
      const double loss = train_step(*model, opt, b, lr, drop_rng);
      if (step % eval_every == 0 || step == steps || step == 1) {
        const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
        std::printf("step %5d  lr %.5f  loss %.4f  greedy exact-match %5.1f%%  (%.0fs)\n", step, lr, loss,
                    100.0 * exact_match(*model, test_gen, 100), secs);
        std::fflush(stdout);
      }
    }

    BeamOptions bo;
    bo.beam_size = beam;
    bo.alpha = alpha;
    TaskGenerator same_a = test_gen, same_b = test_gen;  // identical streams: greedy and beam see the same 300 examples
    const double acc_greedy = exact_match(*model, same_a, 300), acc_beam = exact_match(*model, same_b, 300, &bo);
    std::printf("\nfinal, 300 fresh examples (same for both): greedy %.1f%%  |  beam %d (alpha %.1f) %.1f%%\n",
                100.0 * acc_greedy, beam, alpha, 100.0 * acc_beam);

    std::printf("\nexamples (beam search):\n");
    for (int i = 0; i < 5; ++i) {
      const Example e = test_gen.next();
      BeamOptions o = bo;
      o.max_len = static_cast<int>(e.src.size()) + 5;
      const DecodeResult r = beam_search(*model, e.src, static_cast<int>(e.src.size()), o);
      auto str = [](const std::vector<int>& v) {
        std::string s;
        for (int x : v) s += std::to_string(x - kFirstToken) + " ";
        return s;
      };
      std::printf("  src: %-26s want: %-26s got: %-26s %s\n", str(e.src).c_str(), str(e.tgt).c_str(),
                  str(r.tokens).c_str(), r.tokens == e.tgt ? "ok" : "WRONG");
    }
    if (!save_path.empty()) {
      model->save(save_path);
      std::printf("\nsaved %s\n", save_path.c_str());
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
