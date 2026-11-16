"""Dano de escrita x leitura quando a resposta ainda NAO esta no raciocinio (controle de vazamento).

Vazamento (aproximado, por string): o prefixo de raciocinio do escritor D (ate b tokens, cortado no </think>)
contem \\boxed{ ou o gabarito normalizado (sem espacos). Medido nos traces de D, problema a problema.
Para b em {1024, 2048, 4096}: dano de escrita/leitura (resposta curta) em todos os traces e so nos sem vazamento.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from dxs.markers import stop_id
from dxs.paths import DATA, HF_CACHE

RUNS = [("qwen3-4b", "math500-n300", "Qwen/Qwen3-4B"), ("qwen3-8b", "math500-n300", "Qwen/Qwen3-8B"),
        ("qwen3-1.7b", "math500-n300", "Qwen/Qwen3-1.7B"), ("qwen3-4b", "gpqa_diamond-n198", "Qwen/Qwen3-4B"),
        ("deepseek-r1-distill-llama-8b", "math500-n300", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")]
SPECS = ["gptq-w3g128", "wanda-0.5", "rtn-w3g128"]
rng = np.random.default_rng(0)
rows = []
for m, ds, repo in RUNS:
    tok = AutoTokenizer.from_pretrained(repo, cache_dir=str(HF_CACHE) if "deepseek" not in repo else "/path/to/hf_cache")
    st = stop_id(tok)
    d = pd.read_parquet(DATA / "traces" / f"{m}__{ds}-s4-b16384.parquet", columns=["pid", "sample", "gold", "output_ids"])
    leak = {}
    for r in d.itertuples():
        ids = [int(x) for x in r.output_ids]
        think = ids[:ids.index(st)] if st in ids else ids
        g = str(r.gold).replace(" ", "")
        for b in (1024, 2048, 4096):
            t = tok.decode(think[:b]).replace(" ", "")
            leak[(r.pid, r.sample, b)] = ("\\boxed{" in t) or (len(g) >= 1 and g in t and not g.isalpha()) or (g.isalpha() and len(g) > 1 and g in t)
    for spec in SPECS:
        q = f"{m}__{spec}"
        def load(t, a):
            f = DATA / "swap" / f"{t}__{ds}-s4-b16384__ans-{a}__short.parquet"
            return pd.read_parquet(f) if f.exists() else None
        DD, QD, DQ = load(m, m), load(q, m), load(m, q)
        if DD is None or QD is None or DQ is None:
            continue
        for b in (1024, 2048, 4096):
            k = ["pid", "sample"]
            x = DD[DD.b == b][k + ["correct"]].merge(QD[QD.b == b][k + ["correct"]], on=k, suffixes=("_dd", "_qd")) \
                .merge(DQ[DQ.b == b][k + ["correct"]].rename(columns={"correct": "correct_dq"}), on=k)
            x["leak"] = [leak[(p, s_, b)] for p, s_ in zip(x.pid, x["sample"])]
            for subset, y in (("todos", x), ("sem vazamento", x[~x.leak])):
                g = y.groupby("pid")[["correct_dd", "correct_qd", "correct_dq"]].mean().to_numpy()
                if len(g) < 20:
                    continue
                bt = g[rng.integers(0, len(g), (2000, len(g)))].mean(1)
                w, rd = bt[:, 0] - bt[:, 1], bt[:, 0] - bt[:, 2]
                rows.append({"run": f"{m}/{ds.split('-')[0]}", "compressor": spec, "b": b, "subset": subset,
                             "frac traces": len(y) / len(x), "D->D": g[:, 0].mean(), "escrita": np.median(w), "leitura": np.median(rd),
                             "escrita-leitura IC": f"[{np.percentile(w - rd, 2.5):+.3f}, {np.percentile(w - rd, 97.5):+.3f}]"})
res = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(res.round(3).to_string(index=False))
nl = res[res.subset == "sem vazamento"]
sig = nl["escrita-leitura IC"].apply(lambda s: float(s[1:].split(",")[0]) > 0)
print(f"\nsem vazamento: escrita > leitura com IC fora de zero em {sig.sum()}/{len(nl)} (run x compressor x b)")
res.to_csv(Path(__file__).resolve().parents[1] / "results" / "leakage_check.csv", index=False)
