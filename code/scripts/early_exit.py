"""Saida antecipada na deteccao de loop: detectar o loop durante a geracao e forcar a resposta ali.

O detector e causal: a cada --stride tokens testa se os ultimos TAIL tokens repetem um periodo p <= MAXP em
>= 90% das posicoes (a mesma definicao de loop do artigo, aplicada ao prefixo gerado ate ali). Como o prefixo
ate a deteccao e identico ao trace gravado (mesmo amostrador), a politica e simulada sobre os traces de 16k:
so a resposta forcada no ponto de deteccao e gerada.

Por trace: detectado? posicao, tokens gastos com a politica (deteccao + resposta) e acerto com a politica.
Traces nao detectados mantem o resultado original. Detectores que disparam em traces que terminariam
corretamente aparecem como falsos positivos.

Uso: python scripts/early_exit.py --traces <arquivo 16k> --model <nome ou checkpoint>
Saida: DXS_DATA/early_exit/<traces>__exit.parquet
"""

from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from dxs.data import final_answer, is_correct
from dxs.engine import make_llm, sampling_params
from dxs.paths import DATA, model_path
from dxs.sampling import SamplerCfg
from loop_hazard import MAXP, TAIL

OUT = DATA / "early_exit"
STRIDE = 256


def detect(ids: np.ndarray, stride: int = STRIDE) -> int | None:
    """Primeira posicao (multipla de stride) em que a cauda do prefixo e periodica; None se nunca."""
    n = len(ids)
    for t in range(TAIL + MAXP, n + 1, stride):
        seg = ids[t - TAIL - MAXP:t]
        tail = seg[MAXP:]
        # janelas deslocadas por p = 1..MAXP: linha p-1 e seg[MAXP-p : MAXP-p+TAIL]
        win = sliding_window_view(seg[:-1], TAIL)[::-1]
        if (win == tail).mean(1).max() >= 0.9:
            return t
    return None


def _det(o):
    return detect(np.asarray(o, dtype=np.int64))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--answer-tokens", type=int, default=768)
    ap.add_argument("--gpu-mem", type=float, default=0.75)
    args = ap.parse_args()

    traces = Path(args.traces)
    out = OUT / f"{traces.stem}__exit.parquet"
    if out.exists():
        print(f"[skip] {out}")
        return
    df = pd.read_parquet(traces, columns=["pid", "sample", "gold", "prompt_ids", "output_ids", "n_out", "finish",
                                          "correct"])
    with Pool(16) as pool:
        det = pool.map(_det, list(df.output_ids), chunksize=8)
    df["detect"] = det
    hit = [i for i, d in enumerate(det) if d is not None]
    df["forced_correct"] = None
    df["ans_tokens"] = 0
    if hit:
        from transformers import AutoTokenizer
        path = model_path(args.model)
        tok = AutoTokenizer.from_pretrained(path)
        close = tok.encode("\n</think>\n\n", add_special_tokens=False)
        jobs = []
        for i in hit:
            pre = [int(x) for x in df.output_ids[i][:det[i]]]
            closed = "</think>" in tok.decode(pre, skip_special_tokens=False)
            jobs.append([int(x) for x in df.prompt_ids[i]] + pre + ([] if closed else close))
        llm = make_llm(path, max_model_len=max(map(len, jobs)) + args.answer_tokens + 16, gpu_mem=args.gpu_mem)
        res = llm.generate([{"prompt_token_ids": j} for j in jobs],
                           sampling_params(SamplerCfg(), n=1, max_tokens=args.answer_tokens))
        fc, at = [], []
        for i, r in zip(hit, res):
            o = r.outputs[0]
            fc.append(bool(o.finish_reason == "stop" and is_correct(final_answer("</think>" + o.text), df.gold[i])))
            at.append(len(o.token_ids))
        df.loc[hit, "forced_correct"] = fc
        df.loc[hit, "ans_tokens"] = at
    d = df.drop(columns=["prompt_ids", "output_ids"])
    d["natural_ok"] = (d.finish == "stop") & d.correct
    on = d.detect.notna()
    d["policy_ok"] = np.where(on, d.forced_correct.fillna(False).astype(bool), d.natural_ok)
    d["policy_tokens"] = np.where(on, d.detect.fillna(0) + d.ans_tokens, d.n_out)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(out)
    print(f"[early-exit] detectados={on.mean():.3f} (falsos positivos: {(on & d.natural_ok).sum()}) "
          f"acc natural={d.natural_ok.mean():.3f} politica={d.policy_ok.mean():.3f} "
          f"tokens natural={d.n_out.sum():,} politica={int(d.policy_tokens.sum()):,}")
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
