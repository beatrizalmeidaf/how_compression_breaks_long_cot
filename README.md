# Supplementary material

Anonymous supplementary material for *How Compression Breaks Long Chain-of-Thought: Collapse, Dilation and
Non-Termination*.

## Contents

| Path | What it is |
|---|---|
| `prereg/2026-10-14_crossover.md` | The pre-registered rule of Section 9 (rule, metrics, held-out set, success criteria). |
| `prereg/2026-10-14_crossover.sha256` | Its SHA-256 hash, fixed before the held-out data were generated. |
| `results/*.csv` | Aggregated results behind every table and figure (one row per model/variant/task). |
| `code/dxs/` | Shared library: data loading and grading, vLLM engine, sampling, paths, compressors (RTN, GPTQ, Wanda, AWQ loading). |
| `code/scripts/` | Experiment and analysis scripts, plus the SLURM job files used to run them. |
| `code/env.sh` | Environment variables (cache, data and checkpoint locations; set them for your machine). |
| `code/requirements.txt` | Package versions used. |

Verify the pre-registration from this directory: `sha256sum -c prereg/2026-10-14_crossover.sha256`.

Raw generations (token-level traces, several GB) are not included because of their size; the scripts below
regenerate them. Comments in the code are partly in Portuguese.

## Reproducing the study

```bash
pip install -r code/requirements.txt
cd code && source env.sh          # edit the paths first
python scripts/download_data.py   # MATH-500, GPQA-Diamond, BIG-Bench Hard
```

Compression is simulated (weights quantized or pruned, then stored in bf16):

```bash
python scripts/make_compressed.py --base Qwen/Qwen3-4B rtn-w4g128 gptq-w3g128 wanda-0.5 awq-official
```

Traces (4 samples per problem, temperature 0.6, top-p 0.95, top-k 20, 16,384 tokens):

```bash
python scripts/gen_traces.py --model Qwen/Qwen3-4B --dataset math500 --n-problems 300 --samples 4 --max-tokens 16384
python scripts/gen_traces.py --model qwen3-4b__gptq-w3g128 --dataset math500 --n-problems 300 --samples 4 --max-tokens 16384
```

## Scripts by section

| Section | Scripts |
|---|---|
| 4 Collapse and dilation (fit, validation, per-problem overlap) | `a1_budget_tax.py`, `fit_validation.py`, `problem_overlap.py`, `loop_hazard.py` |
| 5 How compressed models fail | `failure_mode.py`, `failure_paired.py`, `extend_traces.py`, `extend_analysis.py`, `memory_matched.py`, `robustness_check.py`, `throughput.py` |
| 6 Derivation versus stopping, early exit | `force_answer.py`, `rollback.py`, `early_exit.py`, `early_exit_analysis.py` |
| 7 Reading versus writing | `swap_answer.py`, `leakage_check.py`, `dilation_control.py`, `dense_pairs_analysis.py`, `mix_decode.py`, `mix_report.py` |
| Sampling noise, negative results, pre-registration | `temp_shift.py`, `teacher_force.py`, `probe_loops.py`, `extract_feats.py`, `train_length_critic.py`, `fork_rollouts.py`, `p0_analyze.py`, `xover_report.py` |
| Tables and figures | `paper_tables.py`, `paper_figures.py` |

`*.sbatch` files show the exact command lines and settings of each run.
