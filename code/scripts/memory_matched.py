"""Comprimido maior x denso menor com a MESMA memoria de pesos (Qwen3, MATH-500, 16k).

Memoria dos pesos: camadas lineares do decoder em b bits + escala/zero de 16 bits por grupo; embeddings,
cabeca e normas em bf16. Esparsidade nao estruturada (Wanda) nao reduz memoria em formato denso; esses
modelos entram com a memoria do pai e ficam fora das comparacoes.

Para cada denso menor D, cada variante quantizada Q de um pai maior com mem(Q) <= mem(D): diferenca pareada
de acuracia, de tokens por pedido e de nao-terminacao (bootstrap sobre problemas). Custo de decodificacao
aproximado por GB de pesos x tokens (regime limitado por memoria, um pedido por vez).

Saidas: results/memory_matched.csv, results/memory_frontier.csv, acl/tables/memory_matched.tex,
        acl/tables/memory_note.tex
"""
import re
import zlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd

from dxs.paths import DATA

ROOT = Path(__file__).resolve().parents[1]
T = DATA / "traces"
DS = "math500-n300"
B = 16384
# parametros (bilhoes) das lineares do decoder e de embedding+cabeca, contados do config.json
PARAMS = {"qwen3-0.6b": (0.440, 0.156), "qwen3-1.7b": (1.409, 0.311), "qwen3-4b": (3.633, 0.389),
          "qwen3-8b": (6.946, 1.245), "qwen3-14b": (13.212, 1.556), "qwen3-32b": (31.208, 1.557)}
SIZE = {m: i for i, m in enumerate(PARAMS)}
NB = 2000
rng = np.random.default_rng(0)


def bits(spec):
    if spec in ("dense",) or spec.startswith("wanda"):
        return 16.0
    if spec == "awq-official":
        return 4 + 32 / 128
    m = re.fullmatch(r"(?:rtn|gptq)-w(\d+)g(\d+)", spec)
    b, g = int(m[1]), int(m[2])
    return b + (32 / g if g else 0.0)


def mem_gb(base, spec):
    lin, emb = PARAMS[base]
    return (lin * bits(spec) + emb * 16) / 8


def ds_of(name):
    """Qwen3-32B roda num subconjunto pareavel de 100 dos 300 problemas (scripts/q32.sbatch)."""
    return "math500-n100s" if name.startswith("qwen3-32b") else DS


def load(name):
    d = pd.read_parquet(T / f"{name}__{ds_of(name)}-s4-b{B}.parquet", columns=["pid", "finish", "correct", "n_out"])
    d["ok"] = ((d.finish == "stop") & d.correct).astype(float)
    d["tr"] = (d.finish == "length").astype(float)
    return d.groupby("pid")[["ok", "tr", "n_out"]].mean()


rows = []
for f in sorted(list(T.glob(f"qwen3-*__{DS}-s4-b{B}.parquet")) + list(T.glob(f"qwen3-32b*__math500-n100s-s4-b{B}.parquet"))):
    name = re.sub(r"__math500-n\w+-s4-b\d+\.parquet$", "", f.name)
    if "+" in name:
        continue
    base, spec = (name.split("__") + ["dense"])[:2]
    if base not in PARAMS or spec == "awq-real":   # awq-real: mesma config da AWQ simulada (checagem a parte)
        continue
    g = load(name)
    acc = g.ok.mean()
    rows.append({"model": name, "base": base, "compressor": spec, "weights_gb": mem_gb(base, spec), "acc": acc,
                 "non_term": g.tr.mean(), "tokens_per_query": g.n_out.mean(),
                 "tokens_per_correct": g.n_out.mean() / max(acc, 1e-9),
                 "gb_tokens_per_correct": mem_gb(base, spec) * g.n_out.mean() / max(acc, 1e-9) / 1e3})
fr = pd.DataFrame(rows).sort_values("weights_gb")
fr["pareto"] = fr.acc > fr.acc.cummax().shift(fill_value=-1)   # melhor acuracia que todo modelo menor
fr.to_csv(ROOT / "results" / "memory_frontier.csv", index=False)

pairs = []
for _, d in fr[fr.compressor == "dense"].iterrows():
    D = load(d.model)
    for _, q in fr[(fr.compressor != "dense") & ~fr.compressor.str.startswith("wanda")].iterrows():
        if SIZE[q.base] <= SIZE[d.base] or q.weights_gb > d.weights_gb:
            continue
        Q = load(q.model)
        a, b = D.align(Q, join="inner", axis=0)
        idx = np.random.default_rng(zlib.crc32(f"{d.model}|{q.model}".encode())).integers(0, len(a), (NB, len(a)))   # semente por par
        da = b.ok.to_numpy()[idx].mean(1) - a.ok.to_numpy()[idx].mean(1)
        dt = b.n_out.to_numpy()[idx].mean(1) / a.n_out.to_numpy()[idx].mean(1)
        pairs.append({"dense": d.model, "compressed": q.model, "gb dense": d.weights_gb, "gb compr.": q.weights_gb,
                      "acc dense": d.acc, "acc compr.": q.acc, "d_acc": q.acc - d.acc,
                      "d_acc_lo": np.percentile(da, 2.5), "d_acc_hi": np.percentile(da, 97.5),
                      "tok ratio": q.tokens_per_query / d.tokens_per_query,
                      "tok_lo": np.percentile(dt, 2.5), "tok_hi": np.percentile(dt, 97.5),
                      "nonterm dense": d.non_term, "nonterm compr.": q.non_term,
                      "cost ratio": q.gb_tokens_per_correct / d.gb_tokens_per_correct})
pm = pd.DataFrame(pairs)
pm.to_csv(ROOT / "results" / "memory_matched.csv", index=False)
pd.set_option("display.width", 250)
print(fr.round(3).to_string(index=False))
print(pm.round(3).to_string(index=False))

# tabela e nota para o artigo
NM = {"qwen3-0.6b": "Qwen3-0.6B", "qwen3-1.7b": "Qwen3-1.7B", "qwen3-4b": "Qwen3-4B", "qwen3-8b": "Qwen3-8B",
      "qwen3-14b": "Qwen3-14B", "qwen3-32b": "Qwen3-32B"}
NC = {"awq-official": "AWQ-W4", "gptq-w4g128": "GPTQ-W4", "rtn-w4g128": "RTN-W4", "rtn-w4g0": "RTN-W4 (per-ch.)",
      "gptq-w3g128": "GPTQ-W3", "gptq-w3g64": "GPTQ-W3 (g64)", "rtn-w3g128": "RTN-W3", "rtn-w8g128": "RTN-W8"}


def nm(x):
    b, s = (x.split("__") + [None])[:2]
    return NM[b] + (f" {NC[s]}" if s else "")


L = ["\\begin{tabular}{llrrrrlrrr}", "\\toprule",
     "dense & compressed & GB & GB & $\\acc$ & $\\acc$ & $\\Delta\\acc$ [95\\% CI] & tokens & non-term. & cost \\\\",
     " & & dense & compr. & dense & compr. & & ratio & dense / compr. & ratio \\\\", "\\midrule"]
for _, x in pm.sort_values(["gb dense", "d_acc"], ascending=[True, False]).iterrows():
    L.append(f"{nm(x.dense)} & {nm(x.compressed)} & {x['gb dense']:.1f} & {x['gb compr.']:.1f} & {x['acc dense']:.3f} & "
             f"{x['acc compr.']:.3f} & {x.d_acc:+.3f} [{x.d_acc_lo:+.3f}, {x.d_acc_hi:+.3f}] & {x['tok ratio']:.2f} & "
             f"{x['nonterm dense']:.3f} / {x['nonterm compr.']:.3f} & {x['cost ratio']:.2f} \\\\")
L += ["\\bottomrule", "\\end{tabular}"]
(ROOT / "acl" / "tables" / "memory_matched.tex").write_text("\n".join(L) + "\n")
better = pm[pm.d_acc_lo > 0]
worse = pm[pm.d_acc_hi < 0]
more_tok = pm[pm.tok_lo > 1]
note = (f"Of {len(pm)} pairs, the compressed model is more accurate in {len(better)}, less accurate in {len(worse)} "
        f"and not distinguishable in {len(pm) - len(better) - len(worse)}; it uses more tokens per query in "
        f"{len(more_tok)} (median ratio {pm['tok ratio'].median():.2f})")
(ROOT / "acl" / "tables" / "memory_note.tex").write_text(note + "%\n")
print(note)
