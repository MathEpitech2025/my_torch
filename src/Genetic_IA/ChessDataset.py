import numpy as np
from typing import List, Tuple, Dict
import os
import glob

LABEL_MAP: Dict[str, int] = {
    "Nothing": 0,
    "Check": 1,
    "Checkmate": 2,
    "Stalemate": 3
}

PIECE_TO_VAL: Dict[str, float] = {
    'P': 0.1, 'N': 0.3, 'B': 0.3, 'R': 0.5, 'Q': 0.9, 'K': 1.0,
    'p': -0.1, 'n': -0.3, 'b': -0.3, 'r': -0.5, 'q': -0.9, 'k': -1.0,
    '.': 0.0
}

class ChessUtils:
    @staticmethod
    def fen_to_array(fen: str) -> np.ndarray:
        board_str: str = fen.split(" ")[0]
        board_lines: List[str] = board_str.split("/")
        
        piece_indices = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        flat_board = []
        for row in board_lines:
            for char in row:
                if char.isdigit():
                    flat_board.extend(['.'] * int(char))
                else:
                    flat_board.append(char)
        if len(flat_board) < 64:
             flat_board.extend(['.'] * (64 - len(flat_board)))
        flat_board = flat_board[:64]
        input_vector = np.zeros(64 * 12, dtype=np.float32)
        for i, piece_char in enumerate(flat_board):
            if piece_char in piece_indices:
                piece_offset = piece_indices[piece_char]
                idx = (i * 12) + piece_offset
                input_vector[idx] = 1.0
        return input_vector


class ChessDataset:
    def __init__(self, root_dir: str) -> None:
        self.samples: List[Tuple[str, int]] = []

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Folder {root_dir} does,t exist.")

        if os.path.isfile(root_dir):
            self._load_file(root_dir)
            print(f"Loading single-file dataset: {len(self.samples)} samples from {root_dir}.")
            return

        combined_file = os.path.join(root_dir, "combined_dataset.txt")

        if os.path.exists(combined_file):
            self._load_file(combined_file)
            print(f"Loading combined dataset: {len(self.samples)} samples from {combined_file}.")
            return

        search_path = os.path.join(root_dir, "**", "*.txt")
        all_files = glob.glob(search_path, recursive=True)
        if not all_files:
            raise FileNotFoundError(f"No file found {root_dir}")
        for file_path in all_files:
            if os.path.abspath(file_path) == os.path.abspath(combined_file):
                continue
            self._load_file(file_path)

        print(f"Loading completed : {len(self.samples)} exemple found.")

    def _load_file(self, file_path: str):
        try:
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    label_idx = -1
                    if "Checkmate" in line:
                        label_idx = 2
                    elif "Check" in line:
                        label_idx = 1
                    elif "Stalemate" in line:
                        label_idx = 3
                    elif "Nothing" in line:
                        label_idx = 0
                    if label_idx == -1:
                        parent_folder = os.path.basename(os.path.dirname(file_path)).lower()
                        if "checkmate" in parent_folder:
                            label_idx = 2
                        elif "check" in parent_folder:
                            label_idx = 1
                        elif "nothing" in parent_folder:
                            label_idx = 0
                    if label_idx != -1:
                        self.samples.append((line, label_idx))

        except Exception as e:
            print(f"Error when reading {file_path}: {e}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        fen, label_idx = self.samples[idx]
        try:
            fen_array = ChessUtils.fen_to_array(fen)
            return fen_array, label_idx
        except Exception as e:
            print(f"Error converting FEN: {fen} -> {e}")
            return np.zeros(64, dtype=np.float32), 0
