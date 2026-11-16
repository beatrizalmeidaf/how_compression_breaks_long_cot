"""O comprimido resolve os MESMOS problemas que o denso? Sobreposicao por problema em orcamentos efetivos pareados.

Para cada variante com kappa identificado (results/fit_validation.csv), compara Q no orcamento b_Q com D no orcamento
b_Q / kappa, com b_Q = min(16k, 16k * kappa), de modo que os dois fiquem dentro da grade medida. Por problema,
d_p e q_p sao as fracoes de amostras corretas dentro do orcamento. Quantidades:
    ratio    = E[q] / E[d]              (acuracia relativa agregada, o que alpha resume)
    overlap  = E[min(q, d)] / E[d]      (fracao do acerto do denso que o comprimido tambem tem, problema a problema)
    novel    = E[max(q - d, 0)] / E[q]  (fracao do acerto do comprimido em problemas onde o denso acerta menos)
    rho      = correlacao de Pearson entre q_p e d_p
Se o comprimido resolvesse um subconjunto dos problemas do denso, overlap = ratio e novel = 0. Com 4 amostras
ha ruido: a linha de base e o denso contra ele mesmo (amostras 0-1 x 2-3, mesmo orcamento).

Saidas: results/problem_overlap.csv, acl/tables/problem_overlap.tex, acl/tables/problem_overlap_note.tex
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from dxs.paths import DATA

ROOT = Path(__file__).resolve().parents[1]
T = DATA / "traces"
B = 16384
DS = {"math500": "math500-n300", "gpqa_diamond": "gpqa_diamond-n198", "bbh": "bbh-n300"}


def solved(name, ds, b, samples=None):
    d = pd.read_parquet(T / f"{name}__{ds}-s4-b{B}.parquet", columns=["pid", "sample", "finish", "n_out", "correct"])
    if samples is not None:
        d = d[d["sample"].isin(samples)]
    d["ok"] = (d.finish == "stop") & d.correct & (d.n_out <= b)
    return d.groupby("pid").ok.mean()


def metrics(d, q):
    d, q = d.align(q, join="inner")
    d, q = d.to_numpy(), q.to_numpy()
    return {"ratio": q.mean() / d.mean(), "overlap": np.minimum(q, d).mean() / d.mean(),
            "novel": np.maximum(q - d, 0).mean() / max(q.mean(), 1e-9),
            "rho": float(np.corrcoef(q, d)[0, 1]) if q.std() > 0 and d.std() > 0 else np.nan}


fv = pd.read_csv(ROOT / "results" / "fit_validation.csv")
fv = fv[fv.kappa_identified]
rows, base = [], {}
for _, x in fv.iterrows():
    ds = DS[x.task]
    bq = min(B, B * x.kappa)
    bd = bq / x.kappa
    m = metrics(solved(x.model, ds, bd), solved(f"{x.model}__{x.compressor}", ds, bq))
    if (x.model, ds) not in base:   # ruido: denso contra ele mesmo, metades das amostras
        base[(x.model, ds)] = metrics(solved(x.model, ds, B, [0, 1]), solved(x.model, ds, B, [2, 3]))
    rows.append({"model": x.model, "task": x.task, "compressor": x.compressor, "alpha": x.alpha, "kappa": x.kappa,
                 "b_dense": bd, "b_compr": bq, **m, **{f"self_{k}": v for k, v in base[(x.model, ds)].items()}})
r = pd.DataFrame(rows)
r.to_csv(ROOT / "results" / "problem_overlap.csv", index=False)
pd.set_option("display.width", 250)
print(r.round(3).to_string(index=False))

MT = {"qwen3-1.7b": "Qwen3-1.7B", "qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B", "qwen3-14b": "Qwen3-14B",
      "deepseek-r1-distill-llama-8b": "R1-Llama-8B"}
CT = {"rtn-w8g128": "RTN-W8", "awq-official": "AWQ-W4", "gptq-w4g128": "GPTQ-W4", "rtn-w4g128": "RTN-W4",
      "rtn-w4g0": "RTN-W4 (per-ch.)", "wanda-0.3": "Wanda 30\\%", "wanda-0.4": "Wanda 40\\%",
      "gptq-w3g64": "GPTQ-W3 (g64)", "gptq-w3g128": "GPTQ-W3", "wanda-0.5": "Wanda 50\\%", "rtn-w3g128": "RTN-W3"}
TT = {"math500": "MATH", "gpqa_diamond": "GPQA", "bbh": "BBH"}
L = ["\\begin{tabular}{lllrrrrrrr}", "\\toprule",
     "model & task & compressor & $\\kappa$ & ratio & overlap & novel & $\\rho$ & novel (self) & $\\rho$ (self) \\\\",
     "\\midrule"]
for _, x in r.iterrows():
    L.append(f"{MT[x.model]} & {TT[x.task]} & {CT[x.compressor]} & {x.kappa:.2f} & {x.ratio:.2f} & {x.overlap:.2f} & "
             f"{x.novel:.2f} & {x.rho:.2f} & {x.self_novel:.2f} & {x.self_rho:.2f} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
(ROOT / "acl" / "tables" / "problem_overlap.tex").write_text("\n".join(L) + "\n")
gap = r.ratio - r.overlap
note = (f"Across the {len(r)} variants with an identified $\\kappa$, the compressed model's per-problem success "
        f"covers on median {r.overlap.median():.2f} of the dense model's (range {r.overlap.min():.2f}--{r.overlap.max():.2f}), "
        f"against an aggregate ratio of {r.ratio.median():.2f}; a median of {100 * r.novel.median():.0f}\\% of its "
        f"success falls on problems where the dense model does worse, compared with {100 * r.self_novel.median():.0f}\\% "
        f"between two halves of the dense model's own samples")
(ROOT / "acl" / "tables" / "problem_overlap_note.tex").write_text(note + "%\n")
print(note)
