# INSEC on Qwen2.5-7B

Runs the INSEC attack-string optimization (the `opt` stage only) against
`Qwen/Qwen2.5-7B`. No CodeQL / functional-correctness harness needed — the
training loss uses INSEC's built-in per-CWE vulnerability heuristics.

## What was wired up

- `insec/ModelWrapper.py`: `Qwen25Model` (Qwen2.5 FIM format
  `<|fim_prefix|>…<|fim_suffix|>…<|fim_middle|>`), auto device select
  CUDA → Apple-Silicon MPS → CPU (override with `INSEC_DEVICE` / `INSEC_DTYPE`).
- `insec/ModelWrapper.py::load_model`: any `model_dir` containing `qwen` routes here.
- `config.json`: `opt`-only, `gpu: "no"` (no `nvidia-smi` on macOS),
  `sequential: true` (the launcher otherwise starts all 16 CWEs at once — i.e.
  sixteen ~15 GB model loads in parallel).

## Run (all 16 CWEs, via the launcher)

```
cd scripts
source ../venv/bin/activate            # so the spawned `python` is the venv one
python generic_launch.py --config fig3_main/qwen25_7b/config.json --save_dir ../results/qwen25_7b
```

Results land in `results/qwen25_7b/model_dir/final/Qwen2.5-7B/Qwen2.5-7B/<cwe>/result.json`.

## Run one CWE directly (recommended on a single machine)

```
cd scripts
PYTHONPATH=.. ../venv/bin/python run_opt_on_best_init.py \
  --model_dir Qwen/Qwen2.5-7B --tokenizer Qwen/Qwen2.5-7B \
  --dataset cwe-327_py --dataset_dir ../data_train_val \
  --loss_type bbsoft --optimizer random_pool --attack_type comment \
  --attack_position local_prefix --temp 0.4 --top_p 0.95 \
  --num_gen 64 --pool_size 20 --num_adv_tokens 5 --num_train_epochs 500 \
  --seed 0 --output_dir ../results/qwen25_7b_cwe327
```

Add `--manual random` to skip the (slow) inversion-based smart initialization.

## Memory note

`Qwen/Qwen2.5-7B` is ~15 GB in fp16 — it does not fit in 16 GB of unified memory
alongside macOS, and 4-bit `bitsandbytes` is CUDA-only (unavailable on Apple
Silicon). On a 16 GB Mac it will fall back to CPU/swap and be very slow. To try
the pipeline locally, switch `model_dir`/`tokenizer` to a smaller Qwen2.5 (e.g.
`Qwen/Qwen2.5-0.5B` or `Qwen/Qwen2.5-Coder-1.5B`) — identical code path and FIM
tokens. For real 7B runs, use a CUDA GPU (≥16 GB VRAM).
