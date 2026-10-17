# Train, export and run the model

[← README](../README.md) · [Design notes](../DESIGN.md) · [Official quick start](https://wundernn.io/predictorium/docs/quick_start)

Run every command from the **repository root**. No trained checkpoint or ONNX export is included, so the first three steps have to run in order.

## 1. Environment and data

The scorer uses Python 3.11. Any recent Python 3 should work locally.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

[`requirements.txt`](../requirements.txt) is unpinned. `matplotlib` and `seaborn` are only needed for the EDA.

The data and the organizers' `utils.py` come with the starter pack after you sign up on the challenge site. Place them like this (both are git-ignored):

```text
utils.py              # DataPoint and ScorerStepByStep, imported by solution/solution.py
datasets/
├── train.parquet
└── valid.parquet
```

The scripts rely on the provided column order: `seq_ix`, `step_in_seq`, `need_prediction`, then the 32 features, then `t0` and `t1`.

## 2. (Optional) EDA

```bash
python eda_analysis.py
```

This loads both files into memory, prints the report and overwrites the plots in `eda_outputs/`. The report only goes to the console; `eda_output.txt` is a saved copy from my run.

## 3. Train

```bash
python solution/train_mamba2.py \
  --train-path datasets/train.parquet \
  --epochs 6 --window 128 --batch-size 32 \
  --steps-per-epoch 2500 --eval-steps 300 \
  --n-folds 5 --fold 0 \
  --out-dir checkpoints/mamba2_lob
```

These are the defaults written out in full. The run trains one sequence-grouped fold of `train.parquet` and validates on the held-out fold. It uses CUDA when available and the CPU otherwise. Other options: `--seed`, `--lr`, `--weight-decay`, `--warmup-steps`, `--dropout`, `--d-model`, `--n-layers`, `--ff-mult`, `--grad-clip` and `--max-sequences`. See `--help` for their defaults.

| Output | Contents |
| --- | --- |
| `checkpoints/mamba2_lob/best.pt` | Weights, config, CLI args and validation metrics for the best epoch |
| `checkpoints/mamba2_lob/last.pt` | The same for the final epoch |
| `checkpoints/mamba2_lob/train_summary.json` | Best validation score and checkpoint directory |

For a quick pipeline check, cut `--epochs`, `--steps-per-epoch`, `--eval-steps` and `--warmup-steps`, and set `--max-sequences` to at least `--n-folds`. `--max-sequences` trims the data only after the full Parquet file is loaded, so it doesn't reduce peak memory. Use a separate `--out-dir` for each fold or experiment, because the files are overwritten. The [validation caveats](../DESIGN.md#what-validation-does-and-doesnt-establish) explain why the epoch score isn't the streaming score.

## 4. Export one recurrent step to ONNX

```bash
python solution/export_onnx.py \
  --checkpoint checkpoints/mamba2_lob/best.pt \
  --out-dir solution \
  --exporter legacy
```

This writes `solution/mamba2_step.onnx` and `solution/model_meta.json` (dimensions and target transform). `solution.py` looks for exactly these names next to itself. The default opset is 17.

**Use `--exporter legacy`.** The default, `--exporter auto`, tries the Dynamo exporter first when `onnxscript` is installed. With PyTorch 2.14 that attempt fails while validating `dynamic_shapes`, and the script falls back to the legacy exporter after printing a long error. `--exporter dynamo` raises the same error. The legacy export is the path I verified (see [below](#what-ive-checked)). The model file is about 2.2 MB, with no external tensor-data files.

## 5. Score the streaming model locally

```bash
python solution/solution.py
```

This loads the ONNX model, streams every row of `datasets/valid.parquet` through `PredictionModel` the way the scorer does, and prints the weighted Pearson score for `t0`, for `t1` and their mean.

**Check that `Loaded model: …` is printed.** If ONNX Runtime can't load the model, the wrapper prints the error and returns zeros for every scored step, and you get a meaningless score instead of a crash. If `valid.parquet` is missing, the script exits without scoring.

Inference runs on ONNX Runtime, but `solution.py` imports its feature builder from `mamba2_lob.py`, which imports `torch`. PyTorch is in the scoring image, so this works there too.

## 6. Package a submission

`solution.py` must be at the root of the zip. The scorer provides `utils.py`, so leave it out.

```bash
(cd solution && zip ../submission.zip solution.py mamba2_lob.py mamba2_step.onnx model_meta.json)
unzip -l submission.zip
```

```text
submission.zip
├── solution.py
├── mamba2_lob.py
├── mamba2_step.onnx
└── model_meta.json
```

The scorer allows **1 vCPU, no GPU, 16 GB RAM and 60 minutes** for about 1.5 million rows.

## What I've checked

I don't have the competition data or my original checkpoints any more. Before publishing, I ran the whole pipeline end to end on synthetic Parquet files with the real schema (Python 3.13, PyTorch 2.14 CPU, ONNX Runtime 1.23):

- `train_mamba2.py` trains and writes `best.pt`, `last.pt` and `train_summary.json`.
- `export_onnx.py --exporter legacy` produces a model whose outputs match PyTorch's `step()` to within 5e-7 over 300 streamed steps.
- `solution.py` loads the export and scores through the organizers' `utils.ScorerStepByStep`.
- The zip above, extracted next to a copy of `utils.py`, loads and scores.
- Throughput is about **140 µs per row** on one thread of a desktop CPU, including the scorer loop. That's roughly 3.5 minutes for 1.5 million rows, well inside the limit even if the scoring vCPU is several times slower. I haven't measured it on the scorer itself.

The synthetic data says nothing about predictive quality. It only shows that the code runs.

## Keeping generated files out of Git

`.gitignore` covers `datasets/`, `checkpoints/`, `.venv/`, the exported `solution/*.onnx*` and `solution/model_meta.json`, `submission.zip`, and the starter-pack `utils.py` and `example_solution/`. Check `git status` before committing, especially after re-running the EDA, which rewrites the tracked plots in `eda_outputs/`.
