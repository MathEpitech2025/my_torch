import random
import copy
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional
from TrainingModel import TrainingConfig, train_network
from ChessDataset import ChessDataset, LABEL_MAP
import numpy as np

parent_folder_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_folder_src)
from neural_network import NeuralNetwork, loss_functions, activation_functions

MODEL_INPUT_SIZE = 64 * 12
NUM_CLASSES = len(LABEL_MAP)

@dataclass
class Genome:
    learning_rate: float
    batch_size: int
    hidden_layers: List[int]
    activation_fns: List[str] = field(default_factory=lambda: ["relu"])
    fitness: float = 0.0
    eval_time: float = 0.0
    dropouts: float = None
    optimizer: str = "sgd"
    gradient_clip: float = None
    lr_decay: float = 1.0

    def to_config(self) -> TrainingConfig:
        return TrainingConfig(
            learning_rate=self.learning_rate,
            batch_size=self.batch_size,
            hidden_layers=self.hidden_layers,
            dropout_rate=self.dropouts,
            optimizer=self.optimizer,
            gradient_clip=self.gradient_clip,
            lr_decay=self.lr_decay,
            epochs=3
        )
    
    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict) -> 'Genome':
        hidden_layers = data["hidden_layers"]
        if "activation_fns" in data:
            activation_fns = data["activation_fns"]
        else:
            old_act = data.get("activation_fn", "relu")
            activation_fns = [old_act] * len(hidden_layers)

        if len(activation_fns) != len(hidden_layers):
             while len(activation_fns) < len(hidden_layers):
                 activation_fns.append("relu")
             activation_fns = activation_fns[:len(hidden_layers)]

        return Genome(
            learning_rate=data["learning_rate"],
            batch_size=data["batch_size"],
            hidden_layers=hidden_layers,
            activation_fns=activation_fns,
            fitness=data.get("fitness", 0.0),
            eval_time=data.get("eval_time", 0.0),
            dropouts=data.get("dropouts", 0.0),
            optimizer=data.get("optimizer", "sgd"),
            gradient_clip=data.get("gradient_clip", 0.0),
            lr_decay=data.get("lr_decay", 1.0)
        )

class GeneticOptimizer:
    def __init__(self, mutation_rate: float = 0.2, elite_size: int = 2, save_file: str = "evolution_state.json"):
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.save_file = save_file

    def generate_random_genome(self) -> Genome:
        lr = random.uniform(0.0001, 0.01)
        power = random.randint(6, 11)
        batch = 2 ** power
        num_layers = random.randint(1, 3)
        layers = []
        for _ in range(num_layers):
            neurons = random.choice([32, 64, 128, 256])
            layers.append(neurons)
        acts = [random.choice(["relu", "leaky_relu", "gelu", "tanh", "sigmoid"]) for _ in range(num_layers)]
        dropouts = random.uniform(0.0, 0.2) if num_layers > 1 else 0.0
        optimizer = random.choice(["sgd", "adam"])
        gradient_clip = random.choice([None, 1.0, 5.0, 10.0])
        lr_decay = random.uniform(0.95, 1.0)

        return Genome(
            learning_rate=lr,
            batch_size=batch,
            hidden_layers=layers,
            activation_fns=acts,
            dropouts=dropouts,
            optimizer=optimizer,
            gradient_clip=gradient_clip,
            lr_decay=lr_decay
        )

    def create_initial_population(self, pop_size: int = 10) -> List[Genome]:
        print(f"Genesis: Creating initial population of {pop_size} individuals...")
        return [self.generate_random_genome() for _ in range(pop_size)]

    def _sync_if_gpu(self, model: NeuralNetwork):
        if getattr(model, "uses_gpu", False):
            xp = getattr(model, "xp", None)
            try:
                if xp is not None and hasattr(xp, "cuda"):
                    xp.cuda.Stream.null.synchronize()
            except Exception:
                pass

    def evaluate_population(self, population: List[Genome], dataset: ChessDataset):
        print(f"\nStarting Evaluation of {len(population)} individuals...")
        
        for i, genome in enumerate(population):
            if genome.fitness > 0:
                print(f"Individual {i+1} already evaluated (Fitness: {genome.fitness:.2f}% | Time: {genome.eval_time:.2f}s)")
                continue
            print(f"\nTesting Individual {i+1}/{len(population)}: {genome.hidden_layers} | {genome.activation_fns} | LR: {genome.learning_rate:.5f} | Opt: {genome.optimizer}")
            model = NeuralNetwork(MODEL_INPUT_SIZE, loss_function=loss_functions["cross_entropy"], prefer_gpu=True)

            for layer_size, act_name in zip(genome.hidden_layers, genome.activation_fns):
                model.add_layer(layer_size, activation=activation_functions[act_name], dropout_rate=genome.dropouts, optimizer=genome.optimizer)

            model.add_layer(NUM_CLASSES, activation=activation_functions["softmax"], optimizer=genome.optimizer)
            config = genome.to_config()
            start_time = time.perf_counter()
            try:
                accuracy, _ = train_network(model, dataset, config)
                self._sync_if_gpu(model)
                duration = time.perf_counter() - start_time
                genome.fitness = round(accuracy, 0)
                genome.eval_time = duration
                print(f"--> Score: {accuracy:.2f}% | Time: {duration:.2f}s")
            except Exception as e:
                self._sync_if_gpu(model)
                print(f"--> Death by Error: {e}")
                genome.fitness = 0.0
                genome.eval_time = time.perf_counter() - start_time

    def select_best(self, population: List[Genome]) -> Tuple[List[Genome], List[Genome]]:
        sorted_pop = sorted(population, key=lambda g: (round(g.fitness, 0), -g.eval_time), reverse=True)
        elites = sorted_pop[:self.elite_size]
        print(f"Elites preserved: Top {self.elite_size} with scores {[p.fitness for p in elites]}")
        top_50_percent = int(len(population) * 0.5)
        parents_pool = sorted_pop[:top_50_percent]
        return elites, parents_pool

    def crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        child_lr = random.choice([parent1.learning_rate, parent2.learning_rate])
        child_batch = random.choice([parent1.batch_size, parent2.batch_size])
        child_layers = copy.deepcopy(random.choice([parent1.hidden_layers, parent2.hidden_layers]))
        child_dropouts = random.choice([parent1.dropouts, parent2.dropouts])
        child_optimizer = random.choice([parent1.optimizer, parent2.optimizer])
        child_gradient_clip = random.choice([parent1.gradient_clip, parent2.gradient_clip])
        child_lr_decay = random.choice([parent1.lr_decay, parent2.lr_decay])

        if len(parent1.hidden_layers) == len(parent2.hidden_layers):
            child_acts = []
            for a1, a2 in zip(parent1.activation_fns, parent2.activation_fns):
                child_acts.append(random.choice([a1, a2]))
        else:
             if child_layers == parent1.hidden_layers:
                 child_acts = copy.deepcopy(parent1.activation_fns)
             else:
                 child_acts = copy.deepcopy(parent2.activation_fns)

        return Genome(
            learning_rate=child_lr,
            batch_size=child_batch,
            hidden_layers=child_layers,
            activation_fns=child_acts,
            dropouts=child_dropouts,
            optimizer=child_optimizer,
            gradient_clip=child_gradient_clip,
            lr_decay=child_lr_decay
        )

    def mutate(self, genome: Genome):
        if random.random() < self.mutation_rate:
            genome.learning_rate *= random.uniform(0.8, 1.2)
        if random.random() < self.mutation_rate:
            current_idx = [64, 128, 256, 512, 1024, 2048].index(genome.batch_size) if genome.batch_size in [64, 128, 256, 512, 1024, 2048] else 2
            move = random.choice([-1, 1])
            new_idx = max(0, min(5, current_idx + move))
            genome.batch_size = [64, 128, 256, 512, 1024, 2048][new_idx]
        if random.random() < self.mutation_rate:
            action = random.choice(["change", "add", "remove"])
            if action == "change" and len(genome.hidden_layers) > 0:
                genome.hidden_layers[random.randint(0, len(genome.hidden_layers) - 1)] = random.choice([32, 64, 128, 256])
            elif action == "remove" and len(genome.hidden_layers) > 1:
                genome.hidden_layers.pop()
                genome.activation_fns.pop()
            elif action == "add" and len(genome.hidden_layers) < 4:
                genome.hidden_layers.append(random.choice([32, 64, 128]))
                genome.activation_fns.append(random.choice(["relu", "leaky_relu", "gelu", "tanh", "sigmoid"]))

        if random.random() < self.mutation_rate:
            if len(genome.activation_fns) > 0:
                idx = random.randint(0, len(genome.activation_fns) - 1)
                genome.activation_fns[idx] = random.choice(["relu", "leaky_relu", "gelu", "tanh", "sigmoid"])
        if random.random() < self.mutation_rate:
            genome.dropouts = random.uniform(0.0, 0.2)
        if random.random() < self.mutation_rate:
            genome.optimizer = random.choice(["sgd", "adam"])
        if random.random() < self.mutation_rate:
            genome.gradient_clip = random.choice([None, 1.0, 5.0, 10.0])
        if random.random() < self.mutation_rate:
            genome.lr_decay *= random.uniform(0.95, 1.05)
            genome.lr_decay = max(0.9, min(1.0, genome.lr_decay))

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
            for g in population:
                g.fitness = 0.0
                g.eval_time = 0.0
            print(f"--> Resuming from Generation {generation} (fitness reset to 0 for reevaluation with the new input format)")
            return population, generation
        except Exception as e:
            print(f"Error loading save file: {e}. Starting fresh.")
            return [], 0


if __name__ == "__main__":
    optimizer = GeneticOptimizer(save_file="evolution_state.json")
    print("Loading Chess Dataset...")
    if os.path.exists("dataset"):
         full_dataset = ChessDataset("dataset")
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
