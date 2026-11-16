"""Features para o critico de comprimento: hidden final de D, token emitido e tokens restantes.

Para cada trace de D que TERMINOU (os cortados tem comprimento censurado e ficam de fora), sorteia
posicoes j e guarda:
    h_j  hidden final (pos-norm) que preve output[j]       [H] fp16
    y_j  = output[j]
    R_j  = n_out - j - 1   tokens restantes depois de emitir y_j

Uso: python scripts/extract_feats.py --traces <traces de D> --dense Qwen/Qwen3-4B [--every 16]
Saida: DXS_DATA/critic/<traces>__feats.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from dxs.paths import DATA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from teacher_force import final_hidden, load  # noqa: E402

CRITIC = DATA / "critic"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--dense", default="Qwen/Qwen3-4B")
    ap.add_argument("--every", type=int, default=16, help="1 posicao a cada N (offset aleatorio)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    traces = Path(args.traces)
    out = CRITIC / f"{traces.stem}__feats.npz"
    if out.exists():
        print(f"[skip] {out}")
        return
    df = pd.read_parquet(traces)
    df = df[df.finish == "stop"]
    md = load(args.dense)
    rng = np.random.default_rng(args.seed)
    H, Y, R, P = [], [], [], []
    t0 = time.time()
    for i, r in enumerate(df.itertuples()):
        prompt, output = [int(x) for x in r.prompt_ids], [int(x) for x in r.output_ids]
        n = len(output)
        pos = np.arange(rng.integers(args.every), n, args.every)
        hd, _ = final_hidden(md, torch.tensor(prompt + output, device="cuda"))
        H.append(hd[len(prompt) - 1 + pos].to(torch.float16).cpu().numpy())
        Y.append(np.asarray(output)[pos])
        R.append(n - pos - 1)
        P.extend([r.pid] * len(pos))
        if i % 200 == 0:
            print(f"  {i}/{len(df)} {time.time() - t0:.0f}s", flush=True)
    CRITIC.mkdir(parents=True, exist_ok=True)
    np.savez(out, H=np.concatenate(H), y=np.concatenate(Y), R=np.concatenate(R).astype(np.float32),
             pid=np.asarray(P))
    print(f"[ok] {out}: {sum(len(y) for y in Y):,} posicoes de {len(df)} traces ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
