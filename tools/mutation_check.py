#!/usr/bin/env python3
"""Mutation check: inject one classic Transformer bug at a time into a copy of the sources and confirm the
test-suite (double-precision build) fails. Usage: python3 tools/mutation_check.py [substring ...]"""
import subprocess, shutil, sys, os, tempfile
SRC=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# (name, file, old, new)
M = [
 ("no 1/sqrt(d_k) scaling",              "model.hpp",    "Var scores = scale(t, matmul_nt(t, q, k), inv_sqrt_dk);", "Var scores = matmul_nt(t, q, k);"),
 ("no sqrt(d_model) embedding scale",    "model.hpp",    "std::sqrt(static_cast<Real>(h_.d_model)));\n    return dropout(t, add_rows_const", "Real(1));\n    return dropout(t, add_rows_const"),
 ("residual connection dropped",         "model.hpp",    "layernorm(t, add(t, x, dropout(t, sub, h_.dropout, rng)),", "layernorm(t, add(t, dropout(t, sub, h_.dropout, rng), dropout(t, sub, h_.dropout, rng)),"),
 ("decoder self-attention not causal",   "model.hpp",    "mha(t, L.self, y, y, true, T)", "mha(t, L.self, y, y, false, T)"),
 ("source padding mask ignored (cross)", "model.hpp",    "mha(t, L.cross, y, memory, false, src_len)", "mha(t, L.cross, y, memory, false, memory->val->r)"),
 ("softmax backward: missing centering", "autograd.hpp", "(*s->grad)(i, j) += static_cast<Real>((*out->val)(i, j) * ((*out->grad)(i, j) - dot));", "(*s->grad)(i, j) += static_cast<Real>((*out->val)(i, j) * (*out->grad)(i, j));"),
 ("layernorm backward: no xhat*mean term","autograd.hpp","(dxh - s1 - h[j] * s2)", "(dxh - s1)"),
 ("positional encoding: wrong base",     "model.hpp",    "std::pow(10000.0, 2.0 * i / d_model)", "std::pow(1000.0, 2.0 * i / d_model)"),
 ("PE: sin/cos swapped",                 "model.hpp",    "pe(pos, 2 * i) = static_cast<Real>(std::sin(ang));\n      pe(pos, 2 * i + 1) = static_cast<Real>(std::cos(ang));", "pe(pos, 2 * i) = static_cast<Real>(std::cos(ang));\n      pe(pos, 2 * i + 1) = static_cast<Real>(std::sin(ang));"),
 ("label smoothing spread over V not V-1","autograd.hpp", "off = eps / (V - 1)", "off = eps / V"),
 ("Adam without bias correction",        "optim.hpp",    "const double c1 = 1.0 - std::pow(b1_, t_), c2 = 1.0 - std::pow(b2_, t_);", "const double c1 = 1.0, c2 = 1.0;"),
 ("Adam beta2 = 0.999 instead of 0.98",  "optim.hpp",    "double beta2 = 0.98", "double beta2 = 0.999"),
 ("noam schedule: warmup exponent wrong","optim.hpp",    "std::pow(static_cast<double>(warmup_steps), -1.5)", "std::pow(static_cast<double>(warmup_steps), -1.0)"),
 ("tied output: no grad into embedding", "autograd.hpp", "if (b->grad) gemm_tn(*b->grad, *out->grad, *a->val, true);", "if (false) gemm_tn(*b->grad, *out->grad, *a->val, true);"),
 ("FFN without ReLU",                    "model.hpp",    "Var hid = relu(t, add_bias(t, matmul(t, x, leaf(t, *f.w1)), leaf(t, *f.b1)));", "Var hid = add_bias(t, matmul(t, x, leaf(t, *f.w1)), leaf(t, *f.b1));"),
 ("beam: optimistic bound too tight",    "decode.hpp",   "best_live / length_penalty(limit, opt.alpha)", "best_live / length_penalty(1, opt.alpha)"),
 ("length penalty formula",              "decode.hpp",   "std::pow((5.0 + len) / 6.0, alpha)", "std::pow((4.0 + len) / 6.0, alpha)"),
 ("dropout not rescaled",                "autograd.hpp", "const Real s = Real(1) / (Real(1) - p);", "const Real s = Real(1);"),
 ("cross-attn queries/keys swapped",     "model.hpp",    "mha(t, L.cross, y, memory, false, src_len)", "mha(t, L.cross, memory, y, false, src_len)"),
]
only = sys.argv[1:]  # optional name filters
for name, f, old, new in M:
    if only and not any(o in name for o in only): continue
    d=os.path.join(tempfile.gettempdir(),'aiayn_mut'); shutil.rmtree(d, ignore_errors=True)
    shutil.copytree(SRC+'/include', d+'/include'); shutil.copytree(SRC+'/tests', d+'/tests')
    p=d+'/include/aiayn/'+f; s=open(p).read()
    if old not in s: print(f"!! pattern not found: {name}"); continue
    open(p,'w').write(s.replace(old,new,1))
    c=subprocess.run(['g++','-std=c++23','-O2','-ffp-contract=off','-DAIAYN_REAL=double','-I',d+'/include',d+'/tests/test_all.cpp','-o',d+'/t'],capture_output=True,text=True)
    if c.returncode: print(f"!! compile error: {name}\n{c.stderr[:400]}"); continue
    r=subprocess.run([d+'/t'],capture_output=True,text=True,timeout=300)
    failed=[l[7:] for l in r.stdout.splitlines() if l.startswith('[FAIL]')]
    print(f"{'CAUGHT ' if failed else 'MISSED!'} {name:42s} -> {len(failed)} failing test(s)" + (f"   e.g. {failed[0][:60]}" if failed else ""))
    sys.stdout.flush()
