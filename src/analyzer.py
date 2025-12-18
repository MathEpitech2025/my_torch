import sys
import os
from typing import NoReturn
import pickle

from cli_parser import AnalyzerArgs
from neural_network import NeuralNetwork, load_neuralnetwork

class ChessAnalyzer:

    def __init__(self, args: AnalyzerArgs) -> None:
        self.args: AnalyzerArgs = args
        self.model: NeuralNetwork = self._load_model()

    def _load_model(self) -> NeuralNetwork:

        if not os.path.exists(self.args.load_file):
            raise FileNotFoundError(f"Network file not found: {self.args.load_file}")

        try:
            model = load_neuralnetwork(self.args.load_file)
            if not isinstance(model, NeuralNetwork):
                raise ValueError("Loaded file is not a valid NeuralNetwork instance.")
            return model
        except (pickle.UnpicklingError, EOFError) as e:
            raise ValueError(f"Failed to load neural network (corrupted file?): {e}")

    def run(self) -> None:
        if self.args.train_mode:
            self._run_training()
        elif self.args.predict_mode:
            self._run_prediction()
        else:
            raise ValueError("No mode specified (neither --train nor --predict).")

    def _run_training(self) -> None:
        print(f"Starting training using dataset: {self.args.chess_file}")

    def _run_prediction(self) -> None:

        print(f"Loading predictions for {self.args.chess_file}...", file=sys.stderr)
