# Design notes: the final streaming model

[← README](README.md) · [Run guide](solution/README.md)

These notes describe iteration 06, the code in [`solution/`](solution/). My original planning checklist also proposed auxiliary volatility prediction, ensembling and augmentation. None of those were built; see [the last section](#planned-but-not-implemented).

## Feature construction: 32 → 69

[`mamba2_lob.py`](solution/mamba2_lob.py) has matching PyTorch (training) and NumPy (inference) feature builders.

| Feature block | Dimensions | Construction |
| --- | ---: | --- |
| Raw state | 32 | `p0…p11`, `v0…v11`, `dp0…dp3`, `dv0…dv3` |
| Change in state | 32 | Current raw vector minus the previous raw vector |
| Volume-imbalance proxy | 1 | `(Σv0…v5 − Σv6…v11) / (Σv0…v5 + Σv6…v11 + 1e-6)` (bid minus ask) |
| Spread proxy | 1 | `p6 − p0` (first ask-price feature minus first bid-price feature) |
| Total-volume proxy | 1 | `Σv0…v11` |
| Position | 2 | `sin(step / 1000)` and `cos(step / 1000)` |
| **Total** | **69** | |

The inputs are anonymized and look standardised (every feature lies within ±5.2 in the EDA), so the "volumes" can be negative. That means the imbalance denominator can get close to zero, which a sum of real, non-negative volumes couldn't do. The code adds epsilon but doesn't bound the ratio in any other way.

The first delta is zero at the start of each sampled training window. At inference, the delta is zero only at step 0 of a sequence. Every later step uses the actual previous observation.

## Recurrent architecture

The model projects the 69 features to `d_model = 128` and applies four `Mamba2SSMBlock` layers. Each block runs LayerNorm, a gated recurrence and a residual output projection, followed by a residual feed-forward network (128 → 256 → 128, GELU, dropout 0.1). A final LayerNorm feeds two separate heads (`128 → 64 → 1`) for `t0` and `t1`. With the defaults that comes to about 557k parameters.

For a normalised input `z`, a block projects `z` into an update `u`, a gate `g` and a step size `dt = softplus(·) + 1e-4`, then computes:

```text
a         = exp(-dt × exp(a_log))                 # per-channel decay in (0, 1)
new_state = a × state + (1 - a) × u
y         = sigmoid(g) × new_state + d_skip × z
```

Each block's state is a single vector of width `d_model`. The name refers to Mamba-2's input-dependent decay. There is no convolution, no expanded state dimension and no SSD scan, so this isn't the standard Mamba-2 block.

`forward()` loops over a window and predicts from its last step. `step()` consumes one observation plus explicit states, which is what gets exported to ONNX. The two paths share the same parameters. In my tests, the last-step output of `forward()` matched step-by-step `step()` to about 1e-7.

## Training and checkpoint selection

[`train_mamba2.py`](solution/train_mamba2.py) sorts rows by sequence and step, checks that each sequence has 1,000 rows ordered 0–999, and picks one `GroupKFold` split by `seq_ix`. The defaults are `--n-folds 5 --fold 0`, so one run trains **one fold**, not a five-model ensemble. Only `train.parquet` is used. `valid.parquet` isn't touched during training.

Each batch element samples a training sequence and a target step in 99–999. The window ends at that step, inclusive. Windows that would start before step 0 are left-padded with the first available snapshot, with position set to step 0. The defaults are a 128-step window, batch size 32, 6 epochs of 2,500 optimizer steps, learning rate 2e-4, weight decay 1e-2, 2,000 warm-up steps and gradient clipping at 1.0.

The targets are transformed before computing the loss:

```text
z = sign(y) × log1p(|y|)
```

The loss is the negative weighted Pearson correlation on these transformed targets, with weights `|z|` and predictions clipped to `[-6, 6]` in transformed space, averaged over the two targets. The competition metric uses raw targets for both the weights and the clipping, so the loss is a surrogate, not the scoring objective.

After each epoch, validation samples `--eval-steps` batches of windows from the held-out fold, maps the predictions back with the inverse transform, and computes the weighted Pearson score on the raw scale. The best epoch is saved as `best.pt` and the final epoch as `last.pt`.

## Streaming inference

[`export_onnx.py`](solution/export_onnx.py) exports one `step()` with explicit state inputs and outputs:

```text
Inputs:  feature [batch, 69], state_0 … state_3 [batch, 128]
Outputs: pred [batch, 2],     new_state_0 … new_state_3 [batch, 128]
```

[`solution.py`](solution/solution.py) runs this graph with ONNX Runtime on CPU, using one intra-op and one inter-op thread. It keeps the hidden states and the previous raw observation for each `seq_ix`, and resets them at step 0 or when it sees a new ID. Every snapshot updates the state, including the 99 warm-up steps. The wrapper returns `None` when `need_prediction` is false, and otherwise returns the inverse-transformed `[t0, t1]`. Clipping is left to the scorer.

The cost per scored step is constant, with no context window to replay. The organizer's baseline re-runs its GRU over the last 100 snapshots for every prediction. The state cache keeps an entry for each sequence it has seen, about 2 KB of state each, which is negligible for about 1,500 test sequences.

## What validation does and doesn't establish

- **Sequence separation:** training and validation windows come from disjoint sequence IDs. Sequences are independent, so neighbouring IDs don't leak.
- **Sampled vs full scoring:** epoch validation uses random windows and shares the training RNG. It isn't a fixed sweep over all scored steps. `python solution/solution.py` is the full streaming evaluation on `valid.parquet`.
- **Context mismatch:** training starts every window from a zero state. Inference carries the state through the entire sequence. Window padding and first-step deltas also differ. A good windowed validation score therefore doesn't guarantee a good streaming score.
- **Silent failure:** if the ONNX model fails to load, `solution.py` prints the error and returns zeros for every scored step. A run that finishes isn't proof that the intended model loaded. Check for the `Loaded model:` line.

## Planned but not implemented

Auxiliary volatility prediction, multi-seed or multi-fold ensembling, feature-noise augmentation, temporal mixup and a separate ONNX optimisation pass were in my original plan but aren't in the code. Their effect on score or speed is unknown.
