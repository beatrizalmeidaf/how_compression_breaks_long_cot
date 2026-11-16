"""Resultado da saida antecipada na deteccao de loop (scripts/early_exit.py).

Por variante: acuracia natural x com a politica (diferenca pareada, bootstrap sobre problemas), tokens gastos
(razao politica/natural), fracao de traces em que o detector dispara e falsos positivos (disparos em traces
que terminariam corretamente).

Saidas: results/early_exit.csv, acl/tables/early_exit.tex, acl/tables/early_exit_note.tex
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from dxs.paths import DATA

ROOT = Path(__file__).resolve().parents[1]
SEVERE = {"gptq-w3g128", "gptq-w3g64", "wanda-0.5", "rtn-w3g128"}
rng = np.random.default_rng(0)
rows = []
for f in sorted((DATA / "early_exit").glob("*__exit.parquet")):
    stem = f.name.replace("__exit.parquet", "")
    name, ds = stem.rsplit("__", 1)
    d = pd.read_parquet(f)
    d["det"] = d.detect.notna()
    g = d.groupby("pid").agg(nat=("natural_ok", "sum"), pol=("policy_ok", "sum"), tn=("n_out", "sum"),
                             tp=("policy_tokens", "sum"), n=("natural_ok", "size"))
    A = g.to_numpy(float)
    idx = rng.integers(0, len(A), (2000, len(A)))
    s = A[idx].sum(1)
    dacc = (s[:, 1] - s[:, 0]) / s[:, 4]
    tot = A.sum(0)
    base, spec = (name.split("__") + ["dense"])[:2]
    rows.append({"model": base, "compressor": spec, "task": ds.split("-")[0],
                 "regime": "dense" if spec == "dense" else ("severe" if spec in SEVERE else "mild"),
                 "acc natural": tot[0] / tot[4], "acc policy": tot[1] / tot[4], "d_acc": (tot[1] - tot[0]) / tot[4],
                 "d_acc_lo": np.percentile(dacc, 2.5), "d_acc_hi": np.percentile(dacc, 97.5),
                 "token ratio": tot[3] / tot[2], "fired": d.det.mean(),
                 "false pos.": (d.det & d.natural_ok).mean()})
r = pd.DataFrame(rows)
if not len(r):
    (ROOT / "acl" / "tables" / "early_exit.tex").write_text("\\begin{tabular}{l}\\toprule\n(pending)\\\\\n\\bottomrule\n\\end{tabular}\n")
    (ROOT / "acl" / "tables" / "early_exit_note.tex").write_text("Early exit was not evaluated%\n")
    raise SystemExit("[early-exit] nenhum resultado ainda")
r.to_csv(ROOT / "results" / "early_exit.csv", index=False)
pd.set_option("display.width", 250)
print(r.round(3).to_string(index=False))

MT = {"qwen3-0.6b": "Qwen3-0.6B", "qwen3-1.7b": "Qwen3-1.7B", "qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B",
      "qwen3-14b": "Qwen3-14B", "qwen3-32b": "Qwen3-32B", "deepseek-r1-distill-llama-8b": "R1-Llama-8B",
      "deepseek-r1-distill-qwen-1.5b": "R1-Qwen-1.5B"}
CT = {"rtn-w8g128": "RTN-W8", "awq-official": "AWQ-W4", "gptq-w4g128": "GPTQ-W4", "rtn-w4g128": "RTN-W4",
      "rtn-w4g0": "RTN-W4 (per-ch.)", "wanda-0.3": "Wanda 30\\%", "wanda-0.4": "Wanda 40\\%",
      "gptq-w3g64": "GPTQ-W3 (g64)", "gptq-w3g128": "GPTQ-W3", "wanda-0.5": "Wanda 50\\%", "rtn-w3g128": "RTN-W3",
      "dense": "dense"}
TT = {"math500": "MATH", "gpqa_diamond": "GPQA", "bbh": "BBH"}
r["o"] = r.task.map({"math500": 0, "gpqa_diamond": 1, "bbh": 2})
r["s"] = r.model.map(list(MT).index)
r = r.sort_values(["o", "s", "acc natural"], ascending=[True, True, False])
L = ["\\begin{tabular}{lllrrlrrr}", "\\toprule",
     "task & model & compressor & $\\acc$ natural & $\\acc$ exit & $\\Delta\\acc$ [95\\% CI] & tokens & fired & false pos. \\\\",
     "\\midrule"]
for _, x in r.iterrows():
    L.append(f"{TT[x.task]} & {MT[x.model]} & {CT.get(x.compressor, x.compressor)} & {x['acc natural']:.3f} & "
             f"{x['acc policy']:.3f} & {x.d_acc:+.3f} [{x.d_acc_lo:+.3f}, {x.d_acc_hi:+.3f}] & "
             f"{x['token ratio']:.2f} & {x.fired:.3f} & {x['false pos.']:.3f} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
(ROOT / "acl" / "tables" / "early_exit.tex").write_text("\n".join(L) + "\n")

sev, den = r[r.regime == "severe"], r[r.regime == "dense"]
lp = r[r.model.isin(["qwen3-4b", "qwen3-8b"]) & r.compressor.isin(["rtn-w3g128", "wanda-0.5"])]
g3 = r[r.compressor.str.startswith("gptq-w3")]
pct = lambda x: f"{100 * x:.0f}"
neg = r[r.d_acc < 0]
note = (f"Early exit lowers the point estimate of accuracy in {len(neg)} of the {len(r)} models and variants (by at most "
        f"{-r.d_acc.min():.3f}) and none significantly: no 95\\% interval lies entirely below zero, and the lowest lower "
        f"bound of the per-variant intervals is {r.d_acc_lo.min():+.4f}; these are marginal intervals, not a simultaneous "
        f"bound on the loss across variants, and the absence of a significant loss is not a demonstration of equivalence. "
        f"On the {len(den)} dense models, accuracy changes by at most {den.d_acc.max():+.3f} and the detector fires on at "
        f"most {100 * den.fired.max():.1f}\\% of traces. Under severe compression, accuracy rises in "
        f"{(sev.d_acc_lo > 0).sum()} of {len(sev)} variants while generation uses up to "
        f"{pct(1 - sev['token ratio'].min())}\\% fewer tokens. The gains concentrate in the loop-dominated compressors: "
        f"for Qwen3-4B and 8B with RTN-W3 and Wanda-50\\%, accuracy rises by {lp.d_acc.min():+.3f} to {lp.d_acc.max():+.3f} "
        f"with {pct(1 - lp['token ratio'].max())}--{pct(1 - lp['token ratio'].min())}\\% fewer tokens, whereas GPTQ-W3 "
        f"changes by at most {g3.d_acc.max():+.3f}. False positives, traces on which the detector fires that would "
        f"have ended with a correct answer, are at most {100 * r['false pos.'].max():.1f}\\% of traces")
(ROOT / "acl" / "tables" / "early_exit_note.tex").write_text(note + "%\n")
print(note)
