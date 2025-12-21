# Dataset

## Expected format
- Inputs: FEN lines with class indication (Nothing/Check/Checkmate/Stalemate).
- Encoding: one-hot 768 (64 squares × 12 pieces) via `ChessUtils.fen_to_array`.
- Labels: `LABEL_MAP` in `ChessDataset.py` (4 classes).

## Sources
- Class folders under `dataset/` (e.g., `check/`, `checkmate/`, `nothing/`, `stalemate/`), `.txt` files with FEN and label in line or folder name.

## Combined dataset
- `dataset/combined_dataset.txt`: concatenation/dedup of all `.txt` (723,019 unique samples).
- Loaded first by `ChessDataset`. If missing, falls back to individual files.

## Regenerate combined file (optional)
- Merge all `.txt` files under `dataset/` into `dataset/combined_dataset.txt`, skipping duplicate lines and excluding the combined file itself.
- Ensure the resulting file keeps one sample per line and matches the 768→4 target format.

## Compatibility
- Target model: 768 inputs, 4 outputs. A combined file or checkpoint with old dimensions will mismatch.
- Empty dataset: `ChessDataset` raises an explicit error.
