import sys
import os
from typing import NoReturn, List, Tuple, Dict
import numpy as np
from fen_parser import FENParser
import json
# ... (imports)
from fen_parser import FENParser
import pickle
LABEL_MAP: Dict[str, int] = {
    "Nothing": 0,
    "Check White": 1,
    "Check Black": 2,
    "Checkmate White": 3,
    "Checkmate Black": 4,
    "Stalemate": 5
}

from cli_parser import AnalyzerArgs
from neural_network import NeuralNetwork, load_neuralnetwork

class ChessAnalyzer:

    def __init__(self, args: AnalyzerArgs) -> None:
        self.args: AnalyzerArgs = args
        self.model: NeuralNetwork = self._load_model()

    def _load_config(self) -> Dict:
        default_config = {
            "learning_rate": 0.01,
            "batch_size": 32,
            "epochs": 20,
            "validation_split": 0.1
        }
        config_path = os.path.join(os.path.dirname(__file__), "..", "training_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except Exception as e:
                print(f"Warning: Failed to load config ({e}), using defaults.", file=sys.stderr)
        else:
             print("Using default training configuration (no config file found).", file=sys.stderr)
             
        return default_config

    def _calculate_accuracy(self, data: np.ndarray, targets: np.ndarray) -> float:
        if len(data) == 0:
            return 0.0

        batch_size = 1000
        correct = 0
        total = len(data)
        
        for i in range(0, total, batch_size):
            batch_X = data[i:i+batch_size]
            batch_Y = targets[i:i+batch_size]
            
            output = self.model.feedforward(batch_X, training=False)
            predicted_indices = np.argmax(output, axis=1)
            target_indices = np.argmax(batch_Y, axis=1)
            
            correct += np.sum(predicted_indices == target_indices)
        return (correct / total) * 100.0

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
        print(f"Starting training using dataset: {self.args.chess_file}", file=sys.stderr)

        config = self._load_config()
        LEARNING_RATE = float(config["learning_rate"])
        BATCH_SIZE = int(config["batch_size"])
        EPOCHS = int(config["epochs"])
        VAL_SPLIT = float(config["validation_split"])
        
        print(f"Configuration: LR={LEARNING_RATE}, BS={BATCH_SIZE}, Epochs={EPOCHS}, ValSplit={VAL_SPLIT}", file=sys.stderr)

        print("Loading training data...", file=sys.stderr)
        try:
            raw_data = list(FENParser.parse_file(self.args.chess_file, is_train_mode=True))
        except Exception as e:
            raise ValueError(f"Error parsing training file: {e}")
        
        if not raw_data:
            print("Warning: No training data found in file.", file=sys.stderr)
            return

        print(f"Loaded {len(raw_data)} samples.", file=sys.stderr)

        inputs_list: List[np.ndarray] = []
        targets_list: List[np.ndarray] = []

        output_size = self.model.output_size
        
        for fen_grid, label_str in raw_data:
            if not label_str or label_str not in LABEL_MAP:
                continue
                
            label_idx = LABEL_MAP[label_str]
            
            target_vector = np.zeros((1, output_size))
            if label_idx < output_size:
                target_vector[0, label_idx] = 1
            else:
                continue

            inputs_list.append(fen_grid)
            targets_list.append(target_vector)

        if not inputs_list:
             raise ValueError("No valid labeled data found for training (Check label mapping vs Model output size).")

        X = np.array(inputs_list)
        Y = np.concatenate(targets_list, axis=0)
        
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        
        val_size = int(len(X) * VAL_SPLIT)
        if val_size == 0 and len(X) > 1:
            val_size = 1
            
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        X_train = X[train_indices]
        Y_train = Y[train_indices]
        
        X_val = X[val_indices]
        Y_val = Y[val_indices]
        
        print(f"Data Split: {len(X_train)} Train | {len(X_val)} Validation", file=sys.stderr)
        print(f"Training for {EPOCHS} epochs...", file=sys.stderr)
        
        max_accuracy = 0.0
        
        for epoch in range(EPOCHS):
            train_shuffle = np.arange(len(X_train))
            np.random.shuffle(train_shuffle)
            X_curr = X_train[train_shuffle]
            Y_curr = Y_train[train_shuffle]

            for i in range(0, len(X_curr), BATCH_SIZE):
                batch_X = X_curr[i : i + BATCH_SIZE]
                batch_Y = Y_curr[i : i + BATCH_SIZE]
                self.model.feedforward(batch_X, training=True)
                self.model.backpropagation(batch_Y, LEARNING_RATE)

            print(f"Epoch {epoch+1}/{EPOCHS} completed.", file=sys.stderr)
            
            if (epoch + 1) % 5 == 0:
                 val_acc = self._calculate_accuracy(X_val, Y_val)
                 if val_acc > max_accuracy:
                     max_accuracy = val_acc
                 print(f"Epoch {epoch+1}/{EPOCHS} -> Current Val Accuracy: {val_acc:.2f}% (Max: {max_accuracy:.2f}%)", file=sys.stderr)


        save_path = self.args.save_file if self.args.save_file else self.args.load_file
        print(f"Saving trained model to {save_path}...", file=sys.stderr)
        try:
            self.model.save(save_path)
            print("Training completed successfully.", file=sys.stderr)
        except Exception as e:
            raise IOError(f"Failed to save model to {save_path}: {e}")

    def _run_prediction(self) -> None:

        print(f"Loading predictions for {self.args.chess_file}...", file=sys.stderr)
