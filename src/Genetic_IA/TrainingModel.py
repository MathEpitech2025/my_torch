import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from tqdm import tqdm

from ChessLogic import DynamicChessNet
from ChessDataset import ChessDataset

@dataclass
class TrainingConfig:
    learning_rate: float = 0.001
    batch_size: int = 2048
    epochs: int = 10
    hidden_layers: List[int] = field(default_factory=lambda: [64, 32])
    validation_split: float = 0.2

def calculate_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return (100 * correct / total) if total > 0 else 0.0

def train_network(model: nn.Module, dataset: ChessDataset, config: TrainingConfig) -> float:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Training launch on: {device}")

    dataset_size = len(dataset)
    val_size = int(dataset_size * config.validation_split)
    train_size = dataset_size - val_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)

    print(f"Data split: {train_size} training samples | {val_size} validation samples")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(train_loader)
        val_accuracy = calculate_accuracy(model, val_loader, device)
        
        print(f"Epoch {epoch + 1}/{config.epochs} - Loss: {avg_loss:.4f} - Validation Accuracy: {val_accuracy:.2f}%")

    print("Training completed !")
    final_score = calculate_accuracy(model, val_loader, device)
    return final_score


if __name__ == "__main__":
    try:
        print("Loading dataset...")
        dataset = ChessDataset("./dataset")
        
        config = TrainingConfig(
            learning_rate=0.001,
            batch_size=1024,
            epochs=5,
            hidden_layers=[128, 64],
            validation_split=0.2
        )
        
        model = DynamicChessNet(input_size=64, hidden_layers=config.hidden_layers, output_size=4)
        
        accuracy = train_network(model, dataset, config)
        print(f"Final Model Accuracy: {accuracy:.2f}%")
        torch.save(model.state_dict(), "Chess.pth")
        
    except Exception as e:
        print(f"Error during test: {e}")