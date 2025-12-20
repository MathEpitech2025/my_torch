# Direct Training

## Goal
Train the 768→4 network (Nothing, Check, Checkmate, Stalemate) from the one-hot FEN dataset.

## Prerequisites
- Dataset: `dataset/combined_dataset.txt` (preferred). Otherwise, `.txt` files under `dataset/` are scanned.
- Checkpoint: delete old `Chess.pkl` with incompatible dims (not 768→4).
- Backend: set `prefer_gpu=True/False` when constructing the `NeuralNetwork`.

## Run training
```
python3 src/Genetic_IA/TrainingModel.py
```
- Initial log: CPU/GPU backend, train/val split.
- Per epoch: avg loss, val accuracy, current LR, epoch duration.
- End of loop: saves to `Chess.pkl`, prints accuracy and total duration.

## Key parameters (in `TrainingModel.py`)
- `epochs`: dataset passes.
- `learning_rate`, `lr_decay`
- `batch_size`: 512–2048 recommended on GPU if VRAM allows.
- `hidden_layers`, `dropout_rate`, `optimizer` (sgd/adam), `weight_decay`
- `gradient_clip`: cap gradients (e.g., 5.0).
- `validation_split`: fraction reserved for validation.

## Performance tips
- GPU: increase batch while VRAM holds; avoid host/device copies (validation already optimized).
- Timing: epoch durations include validation; GPU sync for timing exists in genetic flow, here no forced per-batch copies.
- Monitor: use `nvidia-smi` during runs.

## Quick troubleshooting
- Dimension mismatch: ensure dataset yields 768 vectors and checkpoint matches (otherwise recreate).
- Pickle “module”: fixed by neutralizing `xp` during `save`.
- Empty dataset: explicit error if no samples.
