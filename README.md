# my_torch — Technical Overview

Minimal dense NN framework for chess positions encoded as FEN. Input: one-hot 768 (64 squares × 12 pieces). Output: 4 classes (Nothing, Check, Checkmate, Stalemate). Main workflows:
- Direct training (`TrainingModel.py`)
- Genetic hyperparameter search (`GeneticOptimizer.py`)
- Network generation from JSON (`my_torch_generator`)

## Structure
- `src/neural_network.py`: core NN (dense layers, activations, dropout, SGD/Adam, numpy backend).
- `src/Genetic_IA/ChessDataset.py`: FEN loading, 768 encoding, labels, combined dataset support (`dataset/combined_dataset.txt`).
- `src/Genetic_IA/TrainingModel.py`: train/validation loop, per-epoch and total timing.
- `src/Genetic_IA/GeneticOptimizer.py`: genome population (hyperparams), crossover/mutation, selection by rounded fitness and time.
- `src/generator/generator.py` + `my_torch_generator`: build architectures from JSON.
- Example configs: `generator_config_best.json`, `generator_config_titan.json`, `training_config.json`.
- Data: `dataset/` (multiple files) and `dataset/combined_dataset.txt` (~723k unique samples).

## Data & format
- Input: FEN string → one-hot 768 via `ChessUtils.fen_to_array`.
- Labels: `LABEL_MAP` (0 Nothing, 1 Check, 2 Checkmate, 3 Stalemate).
- Combined dataset: `combined_dataset.txt` (one FEN+label per line), loaded first if present.

## Backend
- Pure numpy: no GPU dependency or runtime selection.
- Save/load: pickle the full model directly via `NeuralNetwork.save`.

## How to use
- Training guide: see `docs/TRAINING.md`
- Genetic search: see `docs/GENETIC.md`
- Dataset details: see `docs/DATASET.md`
- Quick commands:
  - Direct training: `python3 src/Genetic_IA/TrainingModel.py`
  - Genetic optimization: `python3 src/Genetic_IA/GeneticOptimizer.py`
  - Generation: `./my_torch_generator <config>.json <count>`

## Performance tips
- Adjust batch size based on RAM; start with 512–1024 and tune.
- Keep dropout modest to avoid underfitting on smaller batches.
- Use Adam for quicker convergence on noisy data.

## Contributors
- Laurent Aliu — laurent.aliu@epitech.eu
- Alexandre Guillaud — alexandre.guillaud@epitech.eu
- Enzo Gallini — enzo.gallini@epitech.eu

## Pitfalls
- Dimension compatibility: model 768→4, dataset must match. Old checkpoints (e.g., 64→3/6) will mismatch.
- Pickle: saved directly with `NeuralNetwork.save`.
- Combined dataset: if missing, loader falls back to per-file mode.
