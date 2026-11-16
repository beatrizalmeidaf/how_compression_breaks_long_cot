"""Baixa os datasets que faltam (MATH-500, MATH train, AIME 24/25, GPQA-Diamond) para parquet em DXS_DATA/datasets.

GSM8K e MMLU ja estao no cache local e sao lidos direto de la.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from datasets import concatenate_datasets, load_dataset

from dxs.paths import DATASETS, HF_CACHE

SOURCES = {
    "math500": lambda: load_dataset("HuggingFaceH4/MATH-500", split="test"),
    "math_train": lambda: concatenate_datasets([
        load_dataset("EleutherAI/hendrycks_math", sub, split="train")
        for sub in ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                    "number_theory", "prealgebra", "precalculus")]),
    "aime24": lambda: load_dataset("HuggingFaceH4/aime_2024", split="train"),
    "aime25": lambda: load_dataset("math-ai/aime25", split="test"),
    "gpqa_diamond": lambda: load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train"),
    "gsm8k": lambda: load_dataset("openai/gsm8k", "main", split="test"),
}


def main(names: list[str]) -> None:
    DATASETS.mkdir(parents=True, exist_ok=True)
    for name in names or SOURCES:
        out = DATASETS / f"{name}.parquet"
        if out.exists():
            print(f"[skip] {name}: {out}")
            continue
        try:
            df = SOURCES[name]().to_pandas()
        except Exception as e:  # dataset gated ou fora do ar: seguir com os demais
            print(f"[FALHOU] {name}: {type(e).__name__}: {e}")
            continue
        df.to_parquet(out)
        print(f"[ok] {name}: {len(df)} linhas -> {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
