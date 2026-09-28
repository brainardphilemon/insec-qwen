# Running INSEC on Qwen2.5-7B on Kaggle

`insec_qwen_kaggle.ipynb` runs the INSEC attack-string optimization (the `opt`
stage) against `Qwen/Qwen2.5-7B` on a Kaggle GPU.

## Setup (once)

1. Create a new Kaggle Notebook and upload `insec_qwen_kaggle.ipynb`
   (File → Import Notebook), **or** point a notebook at this repo.
2. **Settings → Accelerator →** `GPU T4 x2` or `GPU P100` (one 16 GB GPU suffices;
   Qwen2.5-7B is ~15 GB in fp16).
3. **Settings → Internet →** `On` — required for `pip`, `git clone`, and the model download.
4. **Add-ons → Secrets →** add a secret named **`GITHUB_TOKEN`** whose value is a
   GitHub personal-access token with `repo` scope (this repo is private), and
   attach it to the notebook.
5. **Run All.**

## What it does

- Clones this repo, installs the `insec` package (+ the minimal runtime deps the
  `opt` path needs), and creates the empty `insec/secret.py`.
- Downloads `Qwen/Qwen2.5-7B` on first use (~15 GB).
- Runs `scripts/run_qwen_opt.sh` for one Python CWE (`cwe-327_py`) as a demo, then
  prints the optimized injected comment from `results/qwen25_7b/<cwe>/result.json`.

## Notes

- The demo uses `EPOCHS=100`; the paper setting is `EPOCHS=500`. Run all 16 CWEs
  with `!cd scripts && EPOCHS=500 bash run_qwen_opt.sh`.
- Non-Python CWEs use a parser during smart-init (cpp→gcc, js→node, go→gofmt);
  install node/go on Kaggle if you run those.
- Results live under `/kaggle/working/insec-qwen/results/` and are saved as the
  notebook Output. Use **Save & Run All (Commit)** for unattended full runs
  (≤ ~9 h per session).
