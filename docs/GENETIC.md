# Genetic Optimization

## Purpose
Automatically explore hyperparameters (LR, batch, layers, activations, dropout, optimizer, clip, decay) and keep the best individuals by accuracy/time.

## Inputs / outputs
- Dataset: `dataset` (uses `combined_dataset.txt` if present).
- Model: built on the fly (768 inputs, 4 outputs).
- State: `evolution_state.json` (population + generation). Fitness/time reset to 0 on reload if format changed.

## Run
```
python3 src/Genetic_IA/GeneticOptimizer.py
```
- Loads state if present, else creates initial population.
- Loop: evaluate → select (elites + parents) → crossover/mutation → save.
- Interrupt: Ctrl+C, rerun to resume.

## Evaluation & timing
- Each individual: short train via `train_network` (epochs=3 by default in `Genome.to_config`).
- Accuracy: rounded to % for ranking.
- Time: GPU synchronized before timing to measure training duration correctly.
- Sorting: by rounded fitness, then time (faster wins ties).

## Key parameters (in `GeneticOptimizer`)
- `mutation_rate`, `elite_size`, population size (via `create_initial_population`).
- Genome: LR ranges, batch (powers of 2), layer counts/sizes, activations, dropout, optimizer, gradient_clip, lr_decay.
- `epochs` per individual: adjust in `Genome.to_config` (heavily impacts generation time).

## Best practices
- GPU: raise `batch_size` if VRAM allows to speed evaluation.
- Large dataset: plan time per generation (~723k samples).
- Incompatible checkpoints: delete/recreate if target architecture changes (stays 768→4 here).

## Resume / reset
- Fresh start: delete `evolution_state.json` or ignore loaded population.
- Re-evaluate with new params: keep the load; fitness is reset automatically if format changed.
