# Benchmark

Comparison of the two evolved models (same architecture; different runs).

| Model | LR | Batch | Hidden | Activations | Dropout | Optimizer | Clip | Decay | Fitness (%) | Eval time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 0.000318845 | 64 | 256 / 256 / 256 | gelu / leaky_relu / tanh | 0.0 | adam | 1.0 | 1.0 | 66.57 | 152.68 |
| B | 0.000318845 | 64 | 256 / 256 / 256 | gelu / leaky_relu / tanh | 0.0 | adam | 1.0 | 1.0 | 66.69 | 177.31 |

Synthetic performance (CPU, random data, batch=64, same architecture):
- Inference: ~12.69 ms/batch (≈5.0k samples/s)
- Train step (forward + backward): ~64.16 ms/batch (≈1.0k samples/s)

Method: random normal inputs shaped (batch, 768) with one-hot labels over 4 classes, timed with `time.perf_counter()` over repeated runs; no GPU.
