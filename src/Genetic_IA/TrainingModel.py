import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from tqdm import tqdm
import os
import sys

from ChessDataset import ChessDataset

parent_folder_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_folder_src)
from neural_network import NeuralNetwork, loss_functions, activation_functions


@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    batch_size: int = 2048
    epochs: int = 10
    hidden_layers: List[int] = field(default_factory=lambda: [64, 32])
    validation_split: float = 0.2

def calculate_accuracy(model: NeuralNetwork, data: List[Tuple[np.ndarray, int]]) -> float:
    correct = 0
    total = len(data)

    for inputs, label in data:
        inputs = inputs.reshape(1, -1)
        output = model.feedforward(inputs)
        predicted = np.argmax(output, axis=1)[0]
        correct += int(predicted == label)

    return (100 * correct / total) if total > 0 else 0.0

def train_network(model: NeuralNetwork, dataset: ChessDataset, config: TrainingConfig) -> float:
    dataset_size = len(dataset)
    val_size = int(dataset_size * config.validation_split)
    train_size = dataset_size - val_size

    indices = np.random.permutation(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_data = [dataset[i] for i in train_indices]
    val_data = [dataset[i] for i in val_indices]

    print(f"Data split: {train_size} training samples | {val_size} validation samples")

    for epoch in range(config.epochs):
        total_loss = 0.0
        np.random.shuffle(train_data)

        num_batches = (len(train_data) + config.batch_size - 1) // config.batch_size
        progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{config.epochs}")

        for batch_idx in progress_bar:
            start_idx = batch_idx * config.batch_size
            end_idx = min(start_idx + config.batch_size, len(train_data))
            batch = train_data[start_idx:end_idx]

            batch_inputs = np.array([item[0] for item in batch])
            batch_labels = np.array([item[1] for item in batch])

            num_classes = model.output_size
            batch_targets = np.zeros((len(batch_labels), num_classes))
            batch_targets[np.arange(len(batch_labels)), batch_labels] = 1

            outputs = model.feedforward(batch_inputs)
            loss = model.loss_function(outputs, batch_targets)
            model.backpropagation(batch_targets, config.learning_rate)

            total_loss += loss
            progress_bar.set_postfix(loss=loss)

        avg_loss = total_loss / num_batches
        val_accuracy = calculate_accuracy(model, val_data)

        print(f"Epoch {epoch + 1}/{config.epochs} - Loss: {avg_loss:.4f} - Validation Accuracy: {val_accuracy:.2f}%")

    print("Training completed !")
    final_score = calculate_accuracy(model, val_data)
    return final_score


if __name__ == "__main__":
    try:
        print("Loading dataset...")
        dataset = ChessDataset("dataset")
        
        config = TrainingConfig(
            learning_rate=0.001,
            batch_size=1024,
            epochs=5,
            hidden_layers=[128, 64],
            validation_split=0.2
        )

        model = NeuralNetwork(64, loss_function=loss_functions["cross_entropy"])
        for layer in config.hidden_layers:
            model.add_layer(layer, activation=activation_functions["relu"])
        model.add_layer(4, activation=activation_functions["softmax"])

        accuracy = train_network(model, dataset, config)
        print(f"Final Model Accuracy: {accuracy:.2f}%")
        model.save("Chess.pkl")
        
    except Exception as e:
        print(f"Error during test: {e}")