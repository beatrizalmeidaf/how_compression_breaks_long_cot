"""Caminhos do servidor. Tudo sobrescrevivel por variavel de ambiente.

`/local` esta quase cheio: dados novos vao para `/raid` (DXS_DATA).
"""

from __future__ import annotations

import os
from pathlib import Path

HF_CACHE = Path(os.environ.get("HF_HUB_CACHE", "/path/to/hf_cache"))
DATA = Path(os.environ.get("DXS_DATA", str(Path.home() / "neuroips_data")))
DATASETS = DATA / "datasets"
TRACES = DATA / "traces"
CKPTS = Path(os.environ.get("DXS_CKPTS", DATA / "ckpts"))   # pesos: grandes e recriaveis
TF = DATA / "tf"
SENS = DATA / "sens"
RESULTS = Path(os.environ.get("DXS_RESULTS", DATA / "results"))

os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE))


def model_path(name: str) -> str:
    """Nome de compressor local (ckpts/<name>) ou repo do HF."""
    local = CKPTS / name
    return str(local) if local.exists() else name
