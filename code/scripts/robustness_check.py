"""Checagens de robustez: kernels reais de 4 bits e um modelo maior (Qwen3-32B), MATH-500, 16k.

(a) AWQ-W4 com kernel real (vLLM, pesos inteiros) x a mesma AWQ simulada (desquantizada para bf16):
    diferenca pareada de acuracia e nao-terminacao; alpha/kappa de cada um contra o denso.
(b) Qwen3-32B: alpha/kappa e mudanca de nao-terminacao (comprimido - pai) por compressor.
Velocidade: results/throughput.csv (scripts/throughput.py), se existir.

Saidas: results/robustness.csv, acl/tables/robustness.tex, acl/tables/robust_kernel_note.tex,
        acl/tables/robust_32b_note.tex, acl/tables/robust_speed_note.tex
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd

from dxs.paths import DATA
from fit_validation import one

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "acl" / "tables"
T = DATA / "traces"
DS = "math500-n300"
B = 16384
CASES = [("qwen3-4b", "awq-official"), ("qwen3-4b", "awq-real"), ("qwen3-8b", "awq-official"), ("qwen3-8b", "awq-real"),
         ("qwen3-32b", "rtn-w4g128"), ("qwen3-32b", "wanda-0.5"), ("qwen3-32b", "rtn-w3g128")]
SEVERE = {"gptq-w3g128", "wanda-0.5", "rtn-w3g128"}
rng = np.random.default_rng(0)


def ds_of(name):
    """Qwen3-32B roda num subconjunto pareavel de 100 dos 300 problemas (scripts/q32.sbatch)."""
    return "math500-n100s" if name.startswith("qwen3-32b") else DS


def f(name):
    return T / f"{name}__{ds_of(name)}-s4-b{B}.parquet"


def per_problem(name):
    d = pd.read_parquet(f(name), columns=["pid", "finish", "correct"])
    d["ok"] = ((d.finish == "stop") & d.correct).astype(float)
    d["tr"] = (d.finish == "length").astype(float)
    d["wr"] = ((d.finish == "stop") & ~d.correct).astype(float)
    return d.groupby("pid")[["ok", "tr", "wr"]].sum().assign(n=d.groupby("pid").size())


def paired(a, b):
    """b - a: acuracia, nao-terminacao, fracao dos erros nao terminados (ponto, lo, hi)."""
    a, b = a.align(b, join="inner", axis=0)
    A, Bm = a[["ok", "tr", "wr", "n"]].to_numpy(float), b[["ok", "tr", "wr", "n"]].to_numpy(float)

    def st(g):
        n = g[..., 3].sum(-1)
        return np.stack([g[..., 0].sum(-1) / n, g[..., 1].sum(-1) / n,
                         g[..., 1].sum(-1) / np.maximum(g[..., 1].sum(-1) + g[..., 2].sum(-1), 1e-9)], -1)
    idx = rng.integers(0, len(A), (2000, len(A)))
    d = st(Bm[idx]) - st(A[idx])
    pt = st(Bm[None])[0] - st(A[None])[0]
    return [(pt[k], *np.percentile(d[:, k], [2.5, 97.5])) for k in range(3)]


if __name__ != "__main__":
    raise SystemExit
from multiprocessing import Pool
todo = [(m, s) for m, s in CASES if f(m).exists() and f(f"{m}__{s}").exists()]
with Pool(max(len(todo), 1)) as pool:
    fits = dict(zip(todo, pool.map(one, [(m, ds_of(m), s) for m, s in todo])))
rows = []
for m, spec in todo:
    r = fits[(m, spec)]
    (da, da_lo, da_hi), (dt, dt_lo, dt_hi), (ds, ds_lo, ds_hi) = paired(per_problem(m), per_problem(f"{m}__{spec}"))
    row = {"model": m, "compressor": spec, "acc": r["acc_Q"], "alpha": r["alpha"], "alpha_lo": r["alpha_lo"],
           "alpha_hi": r["alpha_hi"], "kappa": r["kappa"], "kappa_lo": r["kappa_lo"], "kappa_hi": r["kappa_hi"],
           "d_nonterm": dt, "d_nonterm_lo": dt_lo, "d_nonterm_hi": dt_hi, "d_share": ds, "d_share_lo": ds_lo,
           "d_share_hi": ds_hi}
    if spec == "awq-real" and f(f"{m}__awq-official").exists():
        (ra, ra_lo, ra_hi), (rt, rt_lo, rt_hi), _ = paired(per_problem(f"{m}__awq-official"), per_problem(f"{m}__{spec}"))
        row.update({"real_minus_sim_acc": ra, "rs_acc_lo": ra_lo, "rs_acc_hi": ra_hi,
                    "real_minus_sim_nonterm": rt, "rs_nt_lo": rt_lo, "rs_nt_hi": rt_hi})
    rows.append(row)
res = pd.DataFrame(rows)
res.to_csv(ROOT / "results" / "robustness.csv", index=False)
pd.set_option("display.width", 250)
print(res.round(3).to_string(index=False))

NM = {"qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B", "qwen3-32b": "Qwen3-32B"}
NC = {"awq-official": "AWQ-W4 (simulated)", "awq-real": "AWQ-W4 (real kernel)", "rtn-w4g128": "RTN-W4",
      "gptq-w4g128": "GPTQ-W4", "gptq-w3g128": "GPTQ-W3", "wanda-0.5": "Wanda 50\\%", "rtn-w3g128": "RTN-W3"}
ci = lambda p, lo, hi: f"{p:+.3f} [{lo:+.3f}, {hi:+.3f}]"
L = ["\\begin{tabular}{llrllll}", "\\toprule",
     "model & compressor & $\\acc_Q$ & $\\alpha$ & $\\kappa$ & $\\Delta$ non-term. & $\\Delta$ non-term./err. \\\\",
     "\\midrule"]
for _, x in res.iterrows():
    k = "--" if np.isnan(x.kappa) else f"{x.kappa:.2f} [{x.kappa_lo:.2f}, {x.kappa_hi:.2f}]"
    L.append(f"{NM[x.model]} & {NC[x.compressor]} & {x.acc:.3f} & {x.alpha:.2f} [{x.alpha_lo:.2f}, {x.alpha_hi:.2f}] & "
             f"{k} & {ci(x.d_nonterm, x.d_nonterm_lo, x.d_nonterm_hi)} & {ci(x.d_share, x.d_share_lo, x.d_share_hi)} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
(TAB / "robustness.tex").write_text("\n".join(L) + "\n")

# notas em frases (o texto do artigo as inclui)
real = res[res.compressor == "awq-real"] if len(res) else res
if len(real) and "real_minus_sim_acc" in real:
    parts = [f"{NM[x.model]}: {ci(x.real_minus_sim_acc, x.rs_acc_lo, x.rs_acc_hi)} in accuracy and "
             f"{ci(x.real_minus_sim_nonterm, x.rs_nt_lo, x.rs_nt_hi)} in non-termination" for _, x in real.iterrows()]
    same = all(x.rs_acc_lo <= 0 <= x.rs_acc_hi for _, x in real.iterrows())
    kn = ("Real 4-bit kernels reproduce the simulated results (real $-$ simulated, " if same else
          "Real 4-bit kernels differ from the simulated results in at least one model (real $-$ simulated, ")
    kn += "; ".join(parts) + ")"
else:
    kn = "Real 4-bit kernels were not evaluated"
(TAB / "robust_kernel_note.tex").write_text(kn + "%\n")

big = res[res.model == "qwen3-32b"] if len(res) else res
if len(big):
    sev, mild = big[big.compressor.isin(SEVERE)], big[~big.compressor.isin(SEVERE)]
    up = lambda g: int((g.d_nonterm_lo > 0).sum())
    bn = (f"For Qwen3-32B, severe compression raises non-termination in {up(sev)} of {len(sev)} variants and mild "
          f"compression in {up(mild)} of {len(mild)}; " +
          "; ".join(f"{NC[x.compressor]} reaches {x.acc:.3f} with $\\alpha={x.alpha:.2f}$" +
                    ("" if np.isnan(x.kappa) else f" and $\\kappa={x.kappa:.2f}$") for _, x in big.iterrows()))
else:
    bn = "Qwen3-32B was not evaluated"
(TAB / "robust_32b_note.tex").write_text(bn + "%\n")

tp = ROOT / "results" / "throughput.csv"
if tp.exists():
    t = pd.read_csv(tp).set_index("label")
    sp = []
    for m in ("qwen3-4b", "qwen3-8b"):
        if m in t.index and f"{m}-awq" in t.index:
            sp.append(f"{NM[m]} AWQ-W4 decodes {t.loc[f'{m}-awq', 'tok_s_single'] / t.loc[m, 'tok_s_single']:.2f}$\\times$ "
                      f"as fast as bf16 for a single request and {t.loc[f'{m}-awq', 'tok_s_batch128'] / t.loc[m, 'tok_s_batch128']:.2f}$\\times$ "
                      f"at batch 128")
    sn = "; ".join(sp) if sp else "Decoding speed was not measured"
else:
    sn = "Decoding speed was not measured"
(TAB / "robust_speed_note.tex").write_text(sn + "%\n")
print(kn, bn, sn, sep="\n")
