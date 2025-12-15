import random
import copy
import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional
from TrainingModel import TrainingConfig, train_network
from ChessLogic import DynamicChessNet
from ChessDataset import ChessDataset

@dataclass
class Genome:
    learning_rate: float
    batch_size: int
    hidden_layers: List[int]
    activation_fn: str = "relu"
    fitness: float = 0.0

    def to_config(self) -> TrainingConfig:
        return TrainingConfig(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            hidden_layers=self.hidden_layers,
            epochs=6
        )
    
    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> 'Genome':
        return Genome(
            learning_rate=data["learning_rate"],
            batch_size=data["batch_size"],
            hidden_layers=data["hidden_layers"],
            activation_fn=data.get("activation_fn", "relu"),
            fitness=data.get("fitness", 0.0)
        )

class GeneticOptimizer:
    def __init__(self, mutation_rate: float = 0.2, elite_size: int = 2, save_file: str = "evolution_state.json"):
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.save_file = save_file

    def generate_random_genome(self) -> Genome:
        lr = random.uniform(0.0001, 0.01)
        power = random.randint(5, 11)
        batch = 2 ** power
        num_layers = random.randint(1, 3)
        layers = []
        for _ in range(num_layers):
            neurons = random.choice([32, 64, 128, 256, 512])
            layers.append(neurons)
        act = random.choice(["relu", "leaky_relu", "gelu", "tanh"])
        return Genome(learning_rate=lr, batch_size=batch, hidden_layers=layers, activation_fn=act)

    def create_initial_population(self, pop_size: int = 10) -> List[Genome]:
        print(f"Genesis: Creating initial population of {pop_size} individuals...")
        return [self.generate_random_genome() for _ in range(pop_size)]

    def evaluate_population(self, population: List[Genome], dataset: ChessDataset):
        print(f"\nStarting Evaluation of {len(population)} individuals...")
        
        for i, genome in enumerate(population):
            if genome.fitness > 0:
                print(f"Individual {i+1} already evaluated (Fitness: {genome.fitness:.2f}%)")
                continue
            print(f"\nTesting Individual {i+1}/{len(population)}: {genome.hidden_layers} | {genome.activation_fn} | LR: {genome.learning_rate:.5f}")
            model = DynamicChessNet(
                input_size=64, 
                hidden_layers=genome.hidden_layers, 
                output_size=4,
                activation_fn=genome.activation_fn 
            )
            config = genome.to_config()
            try:
                accuracy = train_network(model, dataset, config)
                genome.fitness = accuracy
                print(f"--> Score: {accuracy:.2f}%")
            except Exception as e:
                print(f"--> Death by Error: {e}")
                genome.fitness = 0.0

    def select_best(self, population: List[Genome]) -> Tuple[List[Genome], List[Genome]]:
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        elites = sorted_pop[:self.elite_size]
        print(f"Elites preserved: Top {self.elite_size} with scores {[p.fitness for p in elites]}")
        top_50_percent = int(len(population) * 0.5)
        parents_pool = sorted_pop[:top_50_percent]
        return elites, parents_pool

    def crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        child_lr = random.choice([parent1.learning_rate, parent2.learning_rate])
        child_batch = random.choice([parent1.batch_size, parent2.batch_size])
        child_layers = copy.deepcopy(random.choice([parent1.hidden_layers, parent2.hidden_layers]))
        child_act = random.choice([parent1.activation_fn, parent2.activation_fn])
        
        return Genome(
            learning_rate=child_lr, 
            batch_size=child_batch, 
            hidden_layers=child_layers,
            activation_fn=child_act
        )

    def mutate(self, genome: Genome):
        if random.random() < self.mutation_rate:
            genome.learning_rate *= random.uniform(0.8, 1.2)
        if random.random() < self.mutation_rate:
            current_idx = [32, 64, 128, 256, 512, 1024, 2048].index(genome.batch_size) if genome.batch_size in [32, 64, 128, 256, 512, 1024, 2048] else 3
            move = random.choice([-1, 1])
            new_idx = max(0, min(6, current_idx + move))
            genome.batch_size = [32, 64, 128, 256, 512, 1024, 2048][new_idx]
        if random.random() < self.mutation_rate:
            action = random.choice(["change", "add", "remove"])
            if action == "change" and len(genome.hidden_layers) > 0:
                genome.hidden_layers[random.randint(0, len(genome.hidden_layers) - 1)] = random.choice([32, 64, 128, 256, 512])
            elif action == "remove" and len(genome.hidden_layers) > 1:
                genome.hidden_layers.pop()
            elif action == "add" and len(genome.hidden_layers) < 4:
                genome.hidden_layers.append(random.choice([32, 64, 128]))
        if random.random() < self.mutation_rate:
            genome.activation_fn = random.choice(["relu", "leaky_relu", "gelu", "tanh"])

    def evolve_generation(self, population: List[Genome], dataset: ChessDataset) -> List[Genome]:
        self.evaluate_population(population, dataset)
        elites, parents_pool = self.select_best(population)
        next_gen = []
        next_gen.extend(elites)
        while len(next_gen) < len(population):
            p1 = random.choice(parents_pool)
            p2 = random.choice(parents_pool)
            child = self.crossover(p1, p2)
            self.mutate(child)
            next_gen.append(child)
            
        print(f"Generation Complete. New population size: {len(next_gen)}")
        return next_gen

    def save_state(self, population: List[Genome], generation: int):
        state = {
            "generation": generation,
            "population": [g.to_dict() for g in population]
        }
        with open(self.save_file, "w") as f:
            json.dump(state, f, indent=4)
        print(f"Evolution saved to {self.save_file} (Gen {generation})")

    def load_state(self) -> Tuple[List[Genome], int]:
        if not os.path.exists(self.save_file):
            return [], 0
            
        print(f"Loading evolution state from {self.save_file}...")
        try:
            with open(self.save_file, "r") as f:
                state = json.load(f)
            population = [Genome.from_dict(g) for g in state["population"]]
            generation = state["generation"]
            print(f"--> Resuming from Generation {generation}")
            return population, generation
        except Exception as e:
            print(f"Error loading save file: {e}. Starting fresh.")
            return [], 0


if __name__ == "__main__":
    optimizer = GeneticOptimizer(save_file="evolution_state.json")
    print("Loading Chess Dataset...")
    if os.path.exists("./dataset"):
         full_dataset = ChessDataset("./dataset")
    else:
         print("No dataset folder found! Using FakeDataset for testing.")
         class FakeDataset:
            def __len__(self): return 100
         full_dataset = FakeDataset()

    population, generation = optimizer.load_state()
    if not population:
        population = optimizer.create_initial_population(pop_size=10)
        generation = 1 
    try:
        while True:
            print(f"\n========================================")
            print(f"GENERATION {generation}")
            print(f"========================================")
            population = optimizer.evolve_generation(population, full_dataset)
            optimizer.save_state(population, generation)
            generation += 1
            
    except KeyboardInterrupt:
        print("\nEvolution paused by user.")
        print(f"To resume, simply run this script again.")
