# my_torch — Technical Overview

Minimal dense NN framework for chess positions encoded as FEN. Input: one-hot 768 (64 squares × 12 pieces). Output: 4 classes (Nothing, Check, Checkmate, Stalemate). Main workflows:
- Direct training (`TrainingModel.py`)
- Genetic hyperparameter search (`GeneticOptimizer.py`)
- Network generation from JSON (`my_torch_generator`)

## Structure
- `src/neural_network.py`: core NN (dense layers, activations, dropout, SGD/Adam, CPU/GPU via `xp`).
- `src/Genetic_IA/ChessDataset.py`: FEN loading, 768 encoding, labels, combined dataset support (`dataset/combined_dataset.txt`).
- `src/Genetic_IA/TrainingModel.py`: train/validation loop, per-epoch and total timing, CPU/GPU logs.
- `src/Genetic_IA/GeneticOptimizer.py`: genome population (hyperparams), crossover/mutation, selection by rounded fitness and time (GPU-synced).
- `src/generator/generator.py` + `my_torch_generator`: build architectures from JSON.
- Example configs: `generator_config_best.json`, `generator_config_titan.json`, `training_config.json`.
- Data: `dataset/` (multiple files) and `dataset/combined_dataset.txt` (~723k unique samples).

## Data & format
- Input: FEN string → one-hot 768 via `ChessUtils.fen_to_array`.
- Labels: `LABEL_MAP` (0 Nothing, 1 Check, 2 Checkmate, 3 Stalemate).
- Combined dataset: `combined_dataset.txt` (one FEN+label per line), loaded first if present.

## CPU/GPU backend
- `prefer_gpu` at model construction picks `cupy` if available, otherwise `numpy`.
- `uses_gpu` reports the current backend. Logs “Backend: GPU/CPU” at train start.
- Save: `NeuralNetwork.save` neutralizes `xp` during pickling to avoid “cannot pickle 'module' object”, then restores (weights moved to CPU for serialization).

## How to use
- Training guide: see `docs/TRAINING.md`
- Genetic search: see `docs/GENETIC.md`
- Dataset details: see `docs/DATASET.md`
- Quick commands:
  - Direct training: `python3 src/Genetic_IA/TrainingModel.py`
  - Genetic optimization: `python3 src/Genetic_IA/GeneticOptimizer.py`
  - Generation: `./my_torch_generator <config>.json <count>`

## Performance tips
- Give work to the GPU: batch 512–2048 if VRAM allows.
- Limit GPU→CPU transfers (validation already optimized).
- Synchronize before timing (already in genetic flow); monitor with `nvidia-smi`.

## Pitfalls
- Dimension compatibility: model 768→4, dataset must match. Old checkpoints (e.g., 64→3/6) will mismatch.
- Pickle: requires neutralizing `xp` during save (fixed).
- Combined dataset: if missing, loader falls back to per-file mode.
