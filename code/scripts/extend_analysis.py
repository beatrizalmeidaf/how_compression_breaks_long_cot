"""Resultado da extensao 16k -> 32k (P1) e resposta forcada no inicio do loop vs em 16k (m1).

Para cada caso: acuracia e nao-terminacao em 16k (todos os traces) e estimativa em 32k
(traces cortados estendidos; loops sorteados entram com peso). Saida: results/extend_32k.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd

from dxs.paths import DATA

ROOT = Path(__file__).resolve().parents[1]
rows = []
for f in sorted((DATA / "extended").glob("*__ext.parquet")):
    stem = f.name.replace("__ext.parquet", "")
    name, ds = stem.rsplit("__", 1)
    full = pd.read_parquet(DATA / "traces" / f"{stem}.parquet", columns=["finish", "correct"])
    e = pd.read_parquet(f)
    N = len(full)
    ok16 = ((full.finish == "stop") & full.correct).sum()
    x = e[e.extended]
    w = x.weight
    gain = (w * x.correct32k.astype(bool)).sum()
    still = (w * (x.finish32k == "length")).sum()
    lp = e[e.loop16k & e.forced_onset.notna()]
    rows.append({"model": name, "task": ds.split("-")[0], "N": N,
                 "acc16k": ok16 / N, "acc32k": (ok16 + gain) / N,
                 "nonterm16k": (full.finish == "length").mean(), "nonterm32k": still / N,
                 "loop share of cut": e.loop16k.mean(),
                 "forced@onset": lp.forced_onset.astype(bool).mean() if len(lp) else float("nan"),
                 "forced@16k": lp.forced_16k.astype(bool).mean() if len(lp) else float("nan"),
                 "n loops forced": len(lp)})
r = pd.DataFrame(rows)
r["gain"] = r.acc32k - r.acc16k
r.to_csv(ROOT / "results" / "extend_32k.csv", index=False)
pd.set_option("display.width", 250)
print(r.round(3).to_string(index=False))

# tabela do apendice
MT = {"qwen3-0.6b": "Qwen3-0.6B", "qwen3-1.7b": "Qwen3-1.7B", "qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B",
      "qwen3-14b": "Qwen3-14B", "deepseek-r1-distill-llama-8b": "R1-Llama-8B", "deepseek-r1-distill-qwen-1.5b": "R1-Qwen-1.5B"}
CT = {"gptq-w3g128": "GPTQ-W3", "gptq-w4g128": "GPTQ-W4", "rtn-w3g128": "RTN-W3", "rtn-w4g0": "RTN-W4 (per-ch.)",
      "wanda-0.3": "Wanda 30\\%", "wanda-0.5": "Wanda 50\\%"}
TT = {"math500": "MATH", "gpqa_diamond": "GPQA", "bbh": "BBH"}
r["base"], r["comp"] = r.model.str.split("__").str[0], r.model.str.split("__").str[1].fillna("dense")
r["o"] = r.task.map({"math500": 0, "gpqa_diamond": 1, "bbh": 2})
r["s"] = r.base.map(list(MT).index)
r = r.sort_values(["o", "s", "acc16k"], ascending=[True, True, False])
L = ["\\begin{tabular}{lllrrrrrr}", "\\toprule",
     "task & model & compressor & $\\acc$ 16k & $\\acc$ 32k & non-term. 16k & non-term. 32k & loops/cut & forced onset / 16k \\\\",
     "\\midrule"]
for _, x in r.iterrows():
    fo = "--" if x["n loops forced"] < 10 else f"{x['forced@onset']:.2f} / {x['forced@16k']:.2f}"
    L.append(f"{TT[x.task]} & {MT[x.base]} & {CT.get(x.comp, 'dense')} & {x.acc16k:.3f} & {x.acc32k:.3f} & "
             f"{x.nonterm16k:.3f} & {x.nonterm32k:.3f} & {x['loop share of cut']:.2f} & {fo} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
(ROOT / "acl" / "tables" / "extend_32k.tex").write_text("\n".join(L) + "\n")
