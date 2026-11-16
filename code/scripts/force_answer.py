"""O conhecimento esta la? Corta o raciocinio em b tokens, forca `</think>` e pede a resposta.

Para cada trace e cada b em --budgets:
    se o trace terminou em ate b tokens  -> usa o resultado dele (nada a forcar)
    senao -> prompt + output[:b] + "\\n</think>\\n\\n" e o MESMO modelo gera a resposta (ate --answer-tokens)
acc_forcada(b) conta os dois casos; o custo e b + tokens da resposta. Comparar com acc(b) sem forcar
(do proprio arquivo de traces) mede quanto da perda e "nao concluir" e quanto e "nao saber".

Uso: python scripts/force_answer.py --traces <traces> --model qwen3-4b__rtn-w3g128
Saida: DXS_DATA/forced/<traces>__forced.parquet  (uma linha por trace x b)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from dxs.data import final_answer, is_correct
from dxs.engine import make_llm, sampling_params
from dxs.paths import DATA, model_path
from dxs.sampling import SamplerCfg

FORCED = DATA / "forced"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--budgets", default="1024,2048,4096,8192,16384")
    ap.add_argument("--answer-tokens", type=int, default=768)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()

    traces = Path(args.traces)
    out = FORCED / f"{traces.stem}__forced.parquet"
    if out.exists():
        print(f"[skip] {out}")
        return
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path(args.model))
    close = tok.encode("\n</think>\n\n", add_special_tokens=False)
    df = pd.read_parquet(traces)
    budgets = [int(b) for b in args.budgets.split(",")]
    rows, jobs = [], []
    for r in df.itertuples():
        out_ids = [int(x) for x in r.output_ids]
        think_open = "</think>" not in r.text
        for b in budgets:
            if r.finish == "stop" and r.n_out <= b:
                rows.append({"pid": r.pid, "sample": r.sample, "b": b, "forced": False, "correct": bool(r.correct), "cost": r.n_out})
            else:
                pre = out_ids[:b]
                # se o </think> ja apareceu antes de b, o modelo ja esta respondendo: so deixa continuar
                closed = "</think>" in tok.decode(pre, skip_special_tokens=False)
                jobs.append((r.pid, r.sample, b, r.gold, pre + ([] if closed else close),
                             [int(x) for x in r.prompt_ids]))
    llm = make_llm(model_path(args.model), max_model_len=max(len(j[5]) + len(j[4]) for j in jobs) + args.answer_tokens + 8,
                   gpu_mem=args.gpu_mem)
    res = llm.generate([{"prompt_token_ids": j[5] + j[4]} for j in jobs],
                       sampling_params(SamplerCfg(), n=1, max_tokens=args.answer_tokens))
    for j, r in zip(jobs, res):
        o = r.outputs[0]
        if o.finish_reason not in ("stop", "length"):
            raise RuntimeError(f"geracao abortada: {o.finish_reason}")
        text = tok.decode(j[4], skip_special_tokens=False) + o.text
        ok = o.finish_reason == "stop" and is_correct(final_answer(text), j[3])
        rows.append({"pid": j[0], "sample": j[1], "b": j[2], "forced": True, "correct": bool(ok),
                     "cost": len(j[4]) + len(o.token_ids)})
    FORCED.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_parquet(out)
    g = res.groupby("b").agg(acc_forcada=("correct", "mean"), frac_forcados=("forced", "mean"), custo=("cost", "mean"))
    print(f"[ok] {out}\n{g.round(4).to_string()}")


if __name__ == "__main__":
    main()
