from os.path import split
import torch
from torch.utils.data import Dataset
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
    'P': 1.0, 'N': 3.0, 'B': 3.0, 'R': 5.0, 'Q': 9.0, 'K': 10.0,
    'p': -1.0, 'n': -3.0, 'b': -3.0, 'r': -5.0, 'q': -9.0, 'k': -10.0,
    '.': 0.0
}

class ChessUtils:
    @staticmethod
    def fen_to_tensor(fen: str) -> torch.Tensor:
        board_str: str = fen.split(" ")[0]
        board_lines: List[str] = board_str.split("/")
        board_values: List[float] = []

        for row in board_lines:
            for char in row:
                if char.isdigit():
                    board_values.extend([0.0] * int(char))
                else:
                    board_values.append(PIECE_TO_VAL.get(char, 0.0))
        if len(board_values) < 64:
            board_values.extend([0.0] * (64 - len(board_values)))
        return torch.tensor(board_values[:64], dtype=torch.float32)


class ChessDataset(Dataset):
    def __init__(self, root_dir: str) -> None:
        self.samples: List[Tuple[str, int]] = []

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Folder {root_dir} does,t exist.")
        search_path = os.path.join(root_dir, "**", "*.txt")
        all_files = glob.glob(search_path, recursive=True)
        if not all_files:
            raise FileNotFoundError(f"No file found {root_dir}")
        for file_path in all_files:
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fen, label_idx = self.samples[idx]
        try:
            fen_tensor = ChessUtils.fen_to_tensor(fen)
            label_tensor = torch.tensor(label_idx, dtype=torch.long)
            return fen_tensor, label_tensor
        except Exception as e:
            print(f"Error converting FEN: {fen} -> {e}")
            return torch.zeros(64, dtype=torch.float32), torch.tensor(0, dtype=torch.long)

