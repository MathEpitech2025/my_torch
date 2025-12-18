import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from tqdm import tqdm
import os
import sys

from ChessDataset import ChessDataset

parent_folder_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_folder_src)
from neural_network import NeuralNetwork, loss_functions, activation_functions, load_neuralnetwork


@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    batch_size: int = 2048
    epochs: int = 10
    hidden_layers: List[int] = field(default_factory=lambda: [64, 32])
    validation_split: float = 0.2
    dropout_rate: float = 0.0
    weight_decay: float = 0.0
    optimizer: str = "sgd"
    gradient_clip: Optional[float] = None
    lr_decay: float = 1.0

def calculate_accuracy(model: NeuralNetwork, data: List[Tuple[np.ndarray, int]]) -> float:
    correct = 0
    total = len(data)

    for inputs, label in data:
        inputs = inputs.reshape(1, -1)
        output = model.feedforward(inputs, training=False)
        predicted = np.argmax(output, axis=1)[0]
        correct += int(predicted == label)

    return (100 * correct / total) if total > 0 else 0.0

def train_network(model: NeuralNetwork, dataset: ChessDataset, config: TrainingConfig) -> Tuple[float, float]:
    dataset_size = len(dataset)
    val_size = int(dataset_size * config.validation_split)
    train_size = dataset_size - val_size

    indices = np.random.permutation(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_data = [dataset[i] for i in train_indices]
    val_data = [dataset[i] for i in val_indices]

    print(f"Data split: {train_size} training samples | {val_size} validation samples")

    current_lr = config.learning_rate
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

            outputs = model.feedforward(batch_inputs, training=True)
            loss = model.loss_function(outputs, batch_targets)
            model.backpropagation(batch_targets, current_lr, config.gradient_clip)

            total_loss += loss
            progress_bar.set_postfix(loss=loss, lr=current_lr)

        avg_loss = total_loss / num_batches
        val_accuracy = calculate_accuracy(model, val_data)

        print(f"Epoch {epoch + 1}/{config.epochs} - Loss: {avg_loss:.4f} - Validation Accuracy: {val_accuracy:.2f}% - LR: {current_lr:.6f}")
        current_lr *= config.lr_decay

    print("Training completed !")
    final_score = calculate_accuracy(model, val_data)
    return final_score, current_lr


if __name__ == "__main__":
    try:
        print("Loading dataset...")
        dataset = ChessDataset("dataset")

        config = TrainingConfig(
            learning_rate=0.001,
            batch_size=64,
            epochs=10,
            hidden_layers=[256, 128, 64],
            validation_split=0.2,
            dropout_rate=0.1,
            weight_decay=0.0,
            optimizer="adam",
            gradient_clip=5.0,
            lr_decay=0.99,
        )

        if os.path.exists("Chess.pkl"):
            print("Loading existing model...")
            model = load_neuralnetwork("Chess.pkl")
        else:
            model = NeuralNetwork(64, loss_function=loss_functions["cross_entropy"])
            for i, layer in enumerate(config.hidden_layers):
                dropout = config.dropout_rate if i < len(config.hidden_layers) - 1 else 0.0
                model.add_layer(
                    layer,
                    activation=activation_functions["leaky_relu"],
                    dropout_rate=dropout,
                    weight_decay=config.weight_decay,
                    optimizer=config.optimizer,
                )
            model.add_layer(3, activation=activation_functions["softmax"], optimizer=config.optimizer)

        for _ in range(100):
            print("Training...")
            accuracy, config.learning_rate = train_network(model, dataset, config)
            print(f"Accuracy: {accuracy:.2f}% | LR: {config.learning_rate:.6f}")
            model.save("Chess.pkl")

    except Exception as e:
        print(f"Error during test: {e}")