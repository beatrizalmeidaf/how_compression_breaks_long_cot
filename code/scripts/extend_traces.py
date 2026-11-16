"""Os traces cortados em 16k convergiriam com mais orcamento? Continua cada trace cortado ate 32k tokens.

Para cada trace com finish == "length":
    (1) extensao: prompt + output (16k) -> o MESMO modelo continua por ate --extra tokens (mesmo amostrador).
        Registra se terminou, quantos tokens usou no total e se acertou.
    (2) resposta forcada no inicio do loop (so traces em loop): prompt + output[:onset] + "</think>" -> resposta.
        Separa "nao derivou" de "o loop destruiu o estado": se forcar no inicio do loop acerta mais do que forcar
        em 16k, o loop apagou uma derivacao que existia.

Uso: python scripts/extend_traces.py --traces <arquivo> --model qwen3-4b__gptq-w3g128
Saida: DXS_DATA/extended/<traces>__ext.parquet (uma linha por trace cortado)
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
from dxs.paths import DATA, model_path
from dxs.sampling import SamplerCfg
from loop_hazard import loop_onset

EXT = DATA / "extended"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--extra", type=int, default=16384)
    ap.add_argument("--answer-tokens", type=int, default=768)
    ap.add_argument("--gpu-mem", type=float, default=0.75)
    ap.add_argument("--max-loops", type=int, default=200, help="estende no maximo N traces em loop (amostra aleatoria)")
    args = ap.parse_args()

    traces = Path(args.traces)
    out = EXT / f"{traces.stem}__ext.parquet"
    if out.exists():
        print(f"[skip] {out}")
        return
    from transformers import AutoTokenizer
    path = model_path(args.model)
    tok = AutoTokenizer.from_pretrained(path)
    close = tok.encode("\n</think>\n\n", add_special_tokens=False)
    df = pd.read_parquet(traces)
    cut = df[df.finish == "length"].reset_index(drop=True)
    if not len(cut):
        print("[nada cortado]")
        return
    prompts = [[int(x) for x in p] for p in cut.prompt_ids]
    outs = [[int(x) for x in o] for o in cut.output_ids]
    onsets = [loop_onset(np.asarray(o)) for o in outs]

    longest = max(len(p) + len(o) for p, o in zip(prompts, outs))
    llm = make_llm(path, max_model_len=longest + args.extra + 16, gpu_mem=args.gpu_mem)

    # (1) extensao: todos os longos sem loop + amostra de ate --max-loops loops (peso = inverso da fracao amostrada)
    loops = [i for i, on in enumerate(onsets) if on is not None]
    rng = np.random.default_rng(0)
    keep = set(rng.choice(loops, min(len(loops), args.max_loops), replace=False).tolist()) if loops else set()
    ext = [i for i in range(len(outs)) if onsets[i] is None or i in keep]
    w_loop = len(loops) / max(len(keep), 1)
    res = dict(zip(ext, llm.generate([{"prompt_token_ids": prompts[i] + outs[i]} for i in ext],
                                     sampling_params(SamplerCfg(), n=1, max_tokens=args.extra))))
    rows = []
    for i in range(len(outs)):
        if i not in res:
            rows.append({"pid": cut.pid[i], "sample": int(cut["sample"][i]), "gold": cut.gold[i], "loop16k": True,
                         "onset": onsets[i], "extended": False, "weight": 0.0})
            continue
        o = res[i].outputs[0]
        if o.finish_reason not in ("stop", "length"):
            raise RuntimeError(f"geracao abortada: {o.finish_reason}")
        text = cut.text[i] + o.text
        ans = final_answer(text) if o.finish_reason == "stop" else None
        full = np.asarray(outs[i] + list(o.token_ids))
        rows.append({"pid": cut.pid[i], "sample": int(cut["sample"][i]), "gold": cut.gold[i],
                     "loop16k": onsets[i] is not None, "onset": onsets[i], "extended": True,
                     "weight": w_loop if onsets[i] is not None else 1.0,
                     "finish32k": o.finish_reason, "n_total": len(outs[i]) + len(o.token_ids),
                     "answer32k": ans, "correct32k": bool(ans is not None and is_correct(ans, cut.gold[i])),
                     "loop32k": o.finish_reason == "length" and loop_onset(full) is not None})

    # (2) resposta forcada no inicio do loop e, para comparar, em 16k (mesmos traces)
    jobs = []
    for i, on in enumerate(onsets):
        if on is None:
            continue
        for where, pre in (("onset", outs[i][:on]), ("16k", outs[i])):
            closed = "</think>" in tok.decode(pre, skip_special_tokens=False)
            jobs.append((i, where, prompts[i] + pre + ([] if closed else close)))
    for r in rows:
        r["forced_onset"] = r["forced_16k"] = None
    if jobs:
        res = llm.generate([{"prompt_token_ids": j[2]} for j in jobs],
                           sampling_params(SamplerCfg(), n=1, max_tokens=args.answer_tokens))
        for (i, where, _), r in zip(jobs, res):
            o = r.outputs[0]
            ok = o.finish_reason == "stop" and is_correct(final_answer("</think>" + o.text), cut.gold[i])
            rows[i][f"forced_{where}"] = bool(ok)

    EXT.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(rows)
    d.to_parquet(out)
    e = d[d.extended]
    for lab, g in (("loops", e[e.loop16k]), ("longos sem loop", e[~e.loop16k])):
        if len(g):
            print(f"[{lab}] n={len(g)} terminam ate 32k={(g.finish32k == 'stop').mean():.3f} "
                  f"corretos ate 32k={g.correct32k.mean():.3f}")
    lp = d[d.loop16k]
    if len(lp):
        print(f"[forcada nos loops] no inicio={lp.forced_onset.mean():.3f} em 16k={lp.forced_16k.mean():.3f}")
    print(f"[ok] {out}")


if __name__ == "__main__":
    main()
