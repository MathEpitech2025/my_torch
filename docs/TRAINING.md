# Direct Training

## Goal
Train the 768→4 network (Nothing, Check, Checkmate, Stalemate) from the one-hot FEN dataset.

## Prerequisites
- Dataset: `dataset/combined_dataset.txt` (preferred). Otherwise, `.txt` files under `dataset/` are scanned.
- Checkpoint: delete old `Chess.pkl` with incompatible dims (not 768→4).

## Run training
```
python3 src/Genetic_IA/TrainingModel.py
```
- Initial log: train/val split.
- Per epoch: avg loss, val accuracy, current LR, epoch duration.
- End of loop: saves to `Chess.pkl`, prints accuracy and total duration.

## Key parameters (in `TrainingModel.py`)
- `epochs`: dataset passes.
- `learning_rate`, `lr_decay`
- `batch_size`: 512–2048 recommended; adjust based on RAM.
- `hidden_layers`, `dropout_rate`, `optimizer` (sgd/adam), `weight_decay`
- `gradient_clip`: cap gradients (e.g., 5.0).
- `validation_split`: fraction reserved for validation.

## Performance tips
- Increase batch if RAM allows; reduce if you hit swapping.
- Adam converges faster on noisy data; SGD is simpler for small batches.
- Keep dropout moderate on small datasets to avoid underfitting.

## Quick troubleshooting
- Dimension mismatch: ensure dataset yields 768 vectors and checkpoint matches (otherwise recreate).
- Pickle: models are saved directly via `NeuralNetwork.save`.
- Empty dataset: explicit error if no samples.
