from typing import Generator, Tuple, Optional, List
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "Genetic_IA"))
from ChessDataset import ChessUtils

class FENParser:
    @staticmethod
    def parse_file(file_path: str, is_train_mode: bool) -> Generator[Tuple[np.ndarray, Optional[str]], None, None]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                clean_line: str = line.strip()
                if not clean_line:
                    continue
                fen: str
                label: Optional[str] = None
                if is_train_mode:
                    parts: List[str] = clean_line.split(" ", 6)
                    if len(parts) > 6:
                        fen = " ".join(parts[:6])
                        label = parts[6].strip()
                    else:
                        fen = clean_line
                        label = None
                else:
                    fen = clean_line

                try:
                    input_array: np.ndarray = ChessUtils.fen_to_array(fen)
                    yield input_array, label
                except Exception:
                    raise ValueError(f"Invalid FEN at line {line_idx}")
