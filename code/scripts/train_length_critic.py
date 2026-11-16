"""Critico de comprimento restante com termo dependente do token:

    Qhat(h, y) = w^T h + c + e_y^T U V^T h          (R em unidades de `scale` tokens)

e_y e a linha do lm_head de D (fixa). So o termo bilinear entra no teacher forcing, porque
sum_y (q - p)(y) [w^T h + c] = 0. Validacao por problema (80/20). Reporta o R^2 do modelo so com
w^T h e o ganho do termo do token: se o ganho for ~0, o critico nao identifica o efeito de y e o
preditor de kappa que depende dele nao tem base.

Uso: python scripts/train_length_critic.py --feats <..._feats.npz> --dense Qwen/Qwen3-4B
Saida: <feats sem _feats>__critic.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from dxs.paths import model_path

SCALE = 1000.0


def r2(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float(1 - ((pred - y) ** 2).mean() / y.var())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--dense", default="Qwen/Qwen3-4B")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    feats = Path(args.feats)
    out = feats.with_name(feats.name.replace("__feats.npz", "__critic.pt"))
    if out.exists():
        print(f"[skip] {out}")
        return
    torch.manual_seed(args.seed)
    z = np.load(feats, allow_pickle=True)
    H = torch.from_numpy(z["H"]).float().cuda()
    y = torch.from_numpy(z["y"]).long().cuda()
    R = torch.from_numpy(z["R"]).float().cuda() / SCALE
    pids = z["pid"]
    uniq = np.unique(pids)
    test_p = set(np.random.default_rng(args.seed).choice(uniq, size=max(1, len(uniq) // 5), replace=False))
    te = torch.from_numpy(np.isin(pids, list(test_p))).cuda()
    tr = ~te

    from transformers import AutoModelForCausalLM
    E = AutoModelForCausalLM.from_pretrained(model_path(args.dense), dtype=torch.bfloat16).lm_head.weight
    E = E.detach().float().cuda()
    E_norm = float(E.norm(dim=1).mean())
    E = E / E_norm
    mu, sd = H[tr].mean(0), H[tr].std(0) + 1e-4
    Hn = (H - mu) / sd
    d = H.shape[1]

    def fit(use_token: bool):
        w = torch.zeros(d, device="cuda", requires_grad=True)
        c = torch.tensor(float(R[tr].mean()), device="cuda", requires_grad=True)
        U = (torch.randn(d, args.rank, device="cuda") * 1e-3).requires_grad_(use_token)
        V = (torch.randn(d, args.rank, device="cuda") * 1e-3).requires_grad_(use_token)
        params = [w, c] + ([U, V] if use_token else [])
        opt = torch.optim.Adam(params, lr=1e-3, weight_decay=args.wd)
        idx_tr = tr.nonzero().squeeze(1)

        def pred(ix):
            base = Hn[ix] @ w + c
            if not use_token:
                return base
            return base + ((Hn[ix] @ V) @ U.T * E[y[ix]]).sum(-1)

        for step in range(args.steps):
            ix = idx_tr[torch.randint(len(idx_tr), (8192,), device="cuda")]
            loss = ((pred(ix) - R[ix]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            ix_te = te.nonzero().squeeze(1)
            return r2(pred(ix_te), R[ix_te]), U.detach(), V.detach()

    r2_base, _, _ = fit(False)
    r2_full, U, V = fit(True)
    # o TF usa h cru: ((h - mu)/sd) V = h V_raw - o. O offset o multiplica e_y, entao NAO cancela em y.
    V_raw = V / sd[:, None]
    o = (mu / sd) @ V
    torch.save({"U": U.cpu(), "V": V_raw.cpu(), "o": o.cpu(), "scale": SCALE, "E_norm": E_norm,
                "r2_base": r2_base, "r2_full": r2_full, "n": int(len(R))}, out)
    print(f"[ok] {out}  R2 teste: so h={r2_base:.4f}  h + token={r2_full:.4f}  ganho do token={r2_full - r2_base:+.4f}")


if __name__ == "__main__":
    main()
