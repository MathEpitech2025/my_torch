import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from tqdm import tqdm
import os
import sys
import time

from ChessDataset import ChessDataset, LABEL_MAP

parent_folder_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_folder_src)
from neural_network import NeuralNetwork, loss_functions, activation_functions, load_neuralnetwork, to_cpu_array

MODEL_INPUT_SIZE = 64 * 12
NUM_CLASSES = len(LABEL_MAP)


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

def calculate_accuracy(model: NeuralNetwork, data: List[Tuple[np.ndarray, int]], batch_size: int = 2048) -> float:
    xp = getattr(model, "xp", np)
    correct = 0
    total = len(data)

    if total == 0:
        return 0.0

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_inputs = xp.asarray([item[0] for item in data[start:end]], dtype=xp.float32)
        batch_labels = xp.asarray([item[1] for item in data[start:end]], dtype=xp.int32)

        outputs = model.feedforward(batch_inputs, training=False)
        predicted = xp.argmax(outputs, axis=1)
        correct += int(to_cpu_array((predicted == batch_labels).sum()))

    return 100 * correct / total

def train_network(model: NeuralNetwork, dataset: ChessDataset, config: TrainingConfig) -> Tuple[float, float]:
    xp = getattr(model, "xp", np)
    print(f"Backend: {'GPU' if getattr(model, 'uses_gpu', False) else 'CPU'} (xp={xp.__name__})")
    if getattr(model, "uses_gpu", False):
        try:
            import cupy as cp
            dev_id = cp.cuda.runtime.getDevice()
            props = cp.cuda.runtime.getDeviceProperties(dev_id)
            print(f"GPU device: {props['name'].decode()} (id={dev_id})")
        except Exception as e:
            print(f"GPU device query failed: {e}")
    dataset_size = len(dataset)
    if dataset_size == 0:
        raise ValueError("Dataset is vide.")

    sample_input, _ = dataset[0]
    feature_dim = len(sample_input)
    if feature_dim != model.input_size:
        raise ValueError(f"Incohérence des dimensions: dataset={feature_dim} features, model={model.input_size}. "
                         f"Le modèle doit être initialisé avec {feature_dim} entrées.")
    val_size = int(dataset_size * config.validation_split)
    train_size = dataset_size - val_size

    indices = np.random.permutation(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_data = [dataset[i] for i in train_indices]
    val_data = [dataset[i] for i in val_indices]

    print(f"Data split: {train_size} training samples | {val_size} validation samples")
    val_batch_size = max(256, min(config.batch_size, 4096))

    current_lr = config.learning_rate
    for epoch in range(config.epochs):
        epoch_start = time.perf_counter()
        total_loss = 0.0
        np.random.shuffle(train_data)

        num_batches = (len(train_data) + config.batch_size - 1) // config.batch_size
        progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch + 1}/{config.epochs}")

        for batch_idx in progress_bar:
            start_idx = batch_idx * config.batch_size
            end_idx = min(start_idx + config.batch_size, len(train_data))
            batch = train_data[start_idx:end_idx]

            batch_inputs = xp.asarray([item[0] for item in batch], dtype=xp.float32)
            batch_labels = xp.asarray([item[1] for item in batch], dtype=xp.int32)

            num_classes = model.output_size
            batch_targets = xp.zeros((len(batch_labels), num_classes), dtype=xp.float32)
            batch_targets[xp.arange(len(batch_labels)), batch_labels] = 1

            outputs = model.feedforward(batch_inputs, training=True)
            loss = model.loss_function(xp, outputs, batch_targets)
            model.backpropagation(batch_targets, current_lr, config.gradient_clip)

            loss_value = float(to_cpu_array(loss))
            total_loss += loss_value
            progress_bar.set_postfix(loss=loss_value, lr=current_lr)

        avg_loss = total_loss / num_batches
        val_accuracy = calculate_accuracy(model, val_data, batch_size=val_batch_size)

        epoch_duration = time.perf_counter() - epoch_start
        print(f"Epoch {epoch + 1}/{config.epochs} - Loss: {avg_loss:.4f} - Validation Accuracy: {val_accuracy:.2f}% - LR: {current_lr:.6f} - Duration: {epoch_duration:.2f}s")
        current_lr *= config.lr_decay

    print("Training completed !")
    final_score = calculate_accuracy(model, val_data, batch_size=val_batch_size)
    return final_score, current_lr


if __name__ == "__main__":
    try:
        print("Loading dataset...")
        dataset_path = "full_training_set.txt" if os.path.exists("full_training_set.txt") else "dataset"
        print(f"Dataset path: {dataset_path}")
        dataset = ChessDataset(dataset_path)

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

        def build_fresh_model():
            net = NeuralNetwork(MODEL_INPUT_SIZE, loss_function=loss_functions["cross_entropy"], prefer_gpu=True)
            for i, layer in enumerate(config.hidden_layers):
                dropout = config.dropout_rate if i < len(config.hidden_layers) - 1 else 0.0
                net.add_layer(
                    layer,
                    activation=activation_functions["leaky_relu"],
                    dropout_rate=dropout,
                    weight_decay=config.weight_decay,
                    optimizer=config.optimizer,
                )
            net.add_layer(NUM_CLASSES, activation=activation_functions["softmax"], optimizer=config.optimizer)
            return net

        if os.path.exists("Chess.pkl"):
            try:
                print("Loading existing model...")
                model = load_neuralnetwork("Chess.pkl", prefer_gpu=True)
                if model.input_size != MODEL_INPUT_SIZE or model.output_size != NUM_CLASSES:
                    print("Ancien modèle incompatible (dimensions). Reconstruction d'un nouveau modèle.")
                    model = build_fresh_model()
            except Exception as e:
                print(f"Impossible de charger l'ancien modèle ({e}), reconstruction.")
                model = build_fresh_model()
        else:
            model = build_fresh_model()

        for _ in range(100):
            print("Training...")
            t0 = time.perf_counter()
            accuracy, config.learning_rate = train_network(model, dataset, config)
            elapsed = time.perf_counter() - t0
            print(f"Accuracy: {accuracy:.2f}% | LR: {config.learning_rate:.6f} | Duration: {elapsed:.2f}s")
            model.save("Chess.pkl")
    except Exception as e:
        print(f"Error during test: {e}")
