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
```python
python3 - <<'PY'
import glob, os
root='dataset'
combined=os.path.join(root,'combined_dataset.txt')
seen=set(); count=0
with open(combined,'w') as out:
    for path in glob.glob(os.path.join(root,'**','*.txt'), recursive=True):
        if os.path.abspath(path)==os.path.abspath(combined):
            continue
        with open(path) as f:
            for line in f:
                line=line.strip()
                if not line or line in seen:
                    continue
                seen.add(line)
                out.write(line+'\\n'); count+=1
print(f\"Wrote {count} unique samples to {combined}\")
PY
```

## Compatibility
- Target model: 768 inputs, 4 outputs. A combined file or checkpoint with old dimensions will mismatch.
- Empty dataset: `ChessDataset` raises an explicit error.
