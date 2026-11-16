"""Intervencao: rollback e reamostragem quando o trace entra em loop.

Parte dos traces JA gerados de Q (rodada 0). Para cada trace em loop:
    volta para `cut` (inicio do loop - margem, ou metade do inicio com --margin half),
    continua gerando com o mesmo sampler e o orcamento que sobrou.
Orcamento honesto: todo token gerado conta, inclusive os descartados. O loop e detectado `--detect`
tokens depois de comecar (idealizacao de um detector online), entao
    gasto = (tokens ate o inicio do loop) + detect   a cada tentativa em loop.
Repete ate --rounds vezes. Saida no formato de traces (finish/n_out/correct), com n_out = gasto total,
para reusar a analise de orcamento: traces/<Q>+rb<margin>__<sufixo>.parquet

Uso: python scripts/rollback.py --traces <traces de Q> --model qwen3-4b__rtn-w3g128 [--margin 64|half]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from dxs.data import final_answer, is_correct
from dxs.engine import make_llm, sampling_params
from dxs.paths import model_path
from dxs.sampling import SamplerCfg
from loop_hazard import loop_onset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--margin", default="64")
    ap.add_argument("--detect", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()

    traces = Path(args.traces)
    stem = traces.stem
    cut_at = stem.rfind("__")
    out = traces.with_name(f"{stem[:cut_at]}+rb{args.margin}{stem[cut_at:]}.parquet")
    if out.exists():
        print(f"[skip] {out}")
        return
    B = int(stem.rsplit("-b", 1)[1])
    df = pd.read_parquet(traces).reset_index(drop=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path(args.model))

    def cut_point(onset: int) -> int:
        return onset // 2 if args.margin == "half" else max(0, onset - int(args.margin))

    # estado por trace: ids atuais, gasto, finish, correto
    ids = [[int(x) for x in o] for o in df.output_ids]
    prompt = [[int(x) for x in p] for p in df.prompt_ids]
    finish = df.finish.tolist()
    correct = df.correct.tolist()
    spent = df.n_out.astype(int).tolist()
    rounds = [0] * len(df)
    pending = {}                                   # idx -> (prefixo mantido, gasto ate aqui)
    for i in range(len(df)):
        if finish[i] == "length":
            o = loop_onset(np.asarray(ids[i]))
            if o is not None:
                pending[i] = (ids[i][:cut_point(o)], o + args.detect)
    n0 = len(pending)
    llm = make_llm(model_path(args.model), max_model_len=max(map(len, prompt)) + B + 16, gpu_mem=args.gpu_mem)
    for rnd in range(1, args.rounds + 1):
        jobs = [(i, pre, s) for i, (pre, s) in pending.items() if B - s > 32]
        for i in set(pending) - {j[0] for j in jobs}:      # sem orcamento: falha definitiva
            spent[i], finish[i], correct[i] = B, "length", False
        pending = {}
        if not jobs:
            break
        res = llm.generate([{"prompt_token_ids": prompt[i] + pre} for i, pre, _ in jobs],
                           [sampling_params(SamplerCfg(), n=1, max_tokens=B - s) for _, _, s in jobs])
        n_ok = 0
        for (i, pre, s), r in zip(jobs, res):
            o = r.outputs[0]
            new = list(o.token_ids)
            rounds[i] = rnd
            full = pre + new
            if o.finish_reason == "stop":
                text = tok.decode(full, skip_special_tokens=False)
                spent[i], finish[i] = s + len(new), "stop"
                correct[i] = is_correct(final_answer(text), df.gold[i])
                n_ok += 1
                continue
            on = loop_onset(np.asarray(full))
            if on is not None and on > len(pre):       # entrou em loop de novo: nova tentativa
                pending[i] = (full[:max(len(pre) // 2 if args.margin == "half" else 0, cut_point(on))],
                              s + (on - len(pre)) + args.detect)
            else:                                       # cortado sem loop (ou loop herdado do prefixo)
                spent[i], finish[i], correct[i] = B, "length", False
        print(f"  rodada {rnd}: {len(jobs)} retomadas, {n_ok} terminaram, {len(pending)} em loop de novo", flush=True)
    for i in pending:
        spent[i], finish[i], correct[i] = B, "length", False

    res = df[["pid", "sample", "gold"]].copy()
    res["finish"], res["n_out"], res["correct"], res["rounds"] = finish, spent, correct, rounds
    res.to_parquet(out)
    print(f"[ok] {out}\n  loops na rodada 0: {n0}/{len(df)}  acc antes={df.correct.mean():.4f} depois={np.mean(correct):.4f} "
          f"cortados antes={(df.finish == 'length').mean():.3f} depois={np.mean([f == 'length' for f in finish]):.3f}")


if __name__ == "__main__":
    main()
