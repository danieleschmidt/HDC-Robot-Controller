"""
Genetic Optimizer for Evolutionary HDC Systems
==============================================

Implements genetic programming and evolutionary strategies for optimizing
hyperdimensional computing algorithms and robot behaviors.
"""

import numpy as np
import random
import time
import copy
from typing import List, Dict, Any, Callable, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import threading
import concurrent.futures
from pathlib import Path
import json

from ..core.hypervector import HyperVector
from ..core.operations import HDCOperations
from ..core.memory import AssociativeMemory


@dataclass
class Individual:
    """Represents an individual in the genetic algorithm."""
    genome: List[Any]
    fitness: float = 0.0
    age: int = 0
    generation: int = 0
    performance_history: List[float] = None
    
    def __post_init__(self):
        if self.performance_history is None:
            self.performance_history = []


@dataclass 
class EvolutionaryParameters:
    """Parameters for evolutionary optimization."""
    population_size: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elitism_ratio: float = 0.1
    tournament_size: int = 5
    max_generations: int = 1000
    convergence_threshold: float = 1e-6
    diversity_preservation: bool = True
    adaptive_parameters: bool = True


class FitnessFunction(ABC):
    """Abstract base class for fitness functions."""
    
    @abstractmethod
    def evaluate(self, individual: Individual, context: Dict[str, Any]) -> float:
        """Evaluate the fitness of an individual."""
        pass
    
    @abstractmethod
    def get_optimal_fitness(self) -> float:
        """Get the optimal fitness value (if known)."""
        pass


class HDCAlgorithmFitness(FitnessFunction):
    """Fitness function for HDC algorithm optimization."""
    
    def __init__(self, test_data: List[Dict], performance_metrics: List[str]):
        self.test_data = test_data
        self.performance_metrics = performance_metrics
        
    def evaluate(self, individual: Individual, context: Dict[str, Any]) -> float:
        """Evaluate HDC algorithm performance."""
        try:
            # Decode genome to HDC parameters
            hdc_params = self._decode_genome(individual.genome)
            
            # Run HDC algorithm with these parameters
            total_score = 0.0
            for test_case in self.test_data:
                score = self._run_test_case(hdc_params, test_case)
                total_score += score
                
            return total_score / len(self.test_data)
            
        except Exception as e:
            # Return low fitness for invalid individuals
            return -1000.0
    
    def get_optimal_fitness(self) -> float:
        return 1.0
        
    def _decode_genome(self, genome: List[Any]) -> Dict[str, Any]:
        """Decode genome to HDC parameters."""
        return {
            'dimension': int(genome[0] * 50000),  # 0-50000
            'learning_rate': genome[1],  # 0-1
            'similarity_threshold': genome[2] * 2 - 1,  # -1 to 1
            'memory_capacity': int(genome[3] * 10000),  # 0-10000
            'bundle_threshold': genome[4],  # 0-1
            'encoding_sparsity': genome[5],  # 0-1
        }
    
    def _run_test_case(self, params: Dict[str, Any], test_case: Dict) -> float:
        """Run a single test case with given parameters."""
        try:
            # Create HDC system with these parameters
            hdc = HDCOperations(params['dimension'])
            memory = AssociativeMemory(params['dimension'])
            
            # Run test and measure performance
            start_time = time.time()
            
            # Simulate learning task
            for i, (input_data, target) in enumerate(test_case['samples']):
                encoded = self._encode_input(input_data, params)
                memory.store(f"sample_{i}", encoded)
                
            # Test recall accuracy
            correct = 0
            for i, (input_data, target) in enumerate(test_case['test_samples']):
                encoded = self._encode_input(input_data, params)
                retrieved = memory.query(encoded, params['similarity_threshold'])
                if retrieved and self._is_correct(retrieved, target):
                    correct += 1
                    
            accuracy = correct / len(test_case['test_samples'])
            latency = time.time() - start_time
            
            # Multi-objective fitness
            fitness = accuracy * 0.7 + (1.0 / (1.0 + latency)) * 0.3
            
            return min(fitness, 1.0)
            
        except Exception:
            return 0.0
    
    def _encode_input(self, input_data: Any, params: Dict[str, Any]) -> HyperVector:
        """Encode input data using HDC parameters."""
        # Simplified encoding - in practice would be more sophisticated
        return HyperVector.random(params['dimension'])
    
    def _is_correct(self, retrieved: str, target: str) -> bool:
        """Check if retrieved result matches target."""
        return retrieved.endswith(target.split('_')[-1])


class GeneticOptimizer:
    """Advanced genetic optimizer for HDC systems."""
    
    def __init__(self, 
                 fitness_function: FitnessFunction,
                 parameters: EvolutionaryParameters = None,
                 seed: int = None):
        self.fitness_function = fitness_function
        self.parameters = parameters or EvolutionaryParameters()
        self.seed = seed
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.population: List[Individual] = []
        self.generation = 0
        self.best_fitness_history = []
        self.average_fitness_history = []
        self.diversity_history = []
        self.is_running = False
        
        # Adaptive parameters
        self.current_mutation_rate = self.parameters.mutation_rate
        self.current_crossover_rate = self.parameters.crossover_rate
        
    def initialize_population(self, genome_length: int) -> List[Individual]:
        """Initialize population with random individuals."""
        population = []
        
        for i in range(self.parameters.population_size):
            genome = [random.random() for _ in range(genome_length)]
            individual = Individual(
                genome=genome,
                generation=0
            )
            population.append(individual)
            
        return population
    
    def evaluate_population(self, population: List[Individual], 
                          context: Dict[str, Any] = None) -> None:
        """Evaluate fitness of entire population in parallel."""
        if context is None:
            context = {}
            
        # Parallel fitness evaluation
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self.fitness_function.evaluate, individual, context): individual 
                for individual in population
            }
            
            for future in concurrent.futures.as_completed(futures):
                individual = futures[future]
                try:
                    fitness = future.result()
                    individual.fitness = fitness
                    individual.performance_history.append(fitness)
                except Exception as e:
                    individual.fitness = -1000.0
    
    def selection(self, population: List[Individual]) -> List[Individual]:
        """Tournament selection for parent selection."""
        selected = []
        
        for _ in range(len(population)):
            # Tournament selection
            tournament = random.sample(population, self.parameters.tournament_size)
            winner = max(tournament, key=lambda ind: ind.fitness)
            selected.append(copy.deepcopy(winner))
            
        return selected
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Uniform crossover between two parents."""
        if random.random() > self.current_crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        genome1 = []
        genome2 = []
        
        for g1, g2 in zip(parent1.genome, parent2.genome):
            if random.random() < 0.5:
                genome1.append(g1)
                genome2.append(g2)
            else:
                genome1.append(g2)
                genome2.append(g1)
        
        child1 = Individual(genome=genome1, generation=self.generation + 1)
        child2 = Individual(genome=genome2, generation=self.generation + 1)
        
        return child1, child2
    
    def mutate(self, individual: Individual) -> Individual:
        """Gaussian mutation with adaptive rate."""
        mutated = copy.deepcopy(individual)
        
        for i in range(len(mutated.genome)):
            if random.random() < self.current_mutation_rate:
                # Gaussian mutation
                mutation_strength = 0.1 * (1.0 - self.generation / self.parameters.max_generations)
                mutation = np.random.normal(0, mutation_strength)
                mutated.genome[i] = max(0.0, min(1.0, mutated.genome[i] + mutation))
        
        mutated.generation = self.generation + 1
        return mutated
    
    def calculate_diversity(self, population: List[Individual]) -> float:
        """Calculate population diversity based on genome distance."""
        if len(population) < 2:
            return 0.0
        
        total_distance = 0.0
        count = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = np.linalg.norm(
                    np.array(population[i].genome) - np.array(population[j].genome)
                )
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def apply_elitism(self, population: List[Individual], 
                     new_population: List[Individual]) -> List[Individual]:
        """Apply elitism to preserve best individuals."""
        elite_count = int(self.parameters.elitism_ratio * len(population))
        
        if elite_count == 0:
            return new_population
        
        # Sort population by fitness
        population.sort(key=lambda ind: ind.fitness, reverse=True)
        elites = population[:elite_count]
        
        # Replace worst individuals in new population with elites
        new_population.sort(key=lambda ind: ind.fitness, reverse=True)
        final_population = elites + new_population[:-elite_count]
        
        return final_population
    
    def adapt_parameters(self):
        """Adapt evolutionary parameters based on progress."""
        if not self.parameters.adaptive_parameters:
            return
        
        if len(self.best_fitness_history) > 10:
            recent_improvement = (self.best_fitness_history[-1] - 
                                self.best_fitness_history[-10]) / 10
            
            if recent_improvement < self.parameters.convergence_threshold:
                # Increase mutation rate if stuck
                self.current_mutation_rate = min(0.5, self.current_mutation_rate * 1.1)
            else:
                # Decrease mutation rate if improving
                self.current_mutation_rate = max(0.01, self.current_mutation_rate * 0.9)
    
    def evolve(self, genome_length: int, 
               context: Dict[str, Any] = None,
               callback: Callable[[int, Dict[str, Any]], None] = None) -> Individual:
        """Run the complete evolutionary optimization process."""
        
        self.is_running = True
        
        # Initialize population
        self.population = self.initialize_population(genome_length)
        
        print(f"🧬 Starting evolutionary optimization with {len(self.population)} individuals")
        
        for generation in range(self.parameters.max_generations):
            if not self.is_running:
                break
                
            self.generation = generation
            
            # Evaluate population
            self.evaluate_population(self.population, context)
            
            # Calculate statistics
            fitnesses = [ind.fitness for ind in self.population]
            best_fitness = max(fitnesses)
            avg_fitness = np.mean(fitnesses)
            diversity = self.calculate_diversity(self.population)
            
            # Store history
            self.best_fitness_history.append(best_fitness)
            self.average_fitness_history.append(avg_fitness)
            self.diversity_history.append(diversity)
            
            # Check convergence
            if best_fitness >= self.fitness_function.get_optimal_fitness() * 0.99:
                print(f"🎯 Converged at generation {generation} with fitness {best_fitness:.6f}")
                break
            
            # Progress callback
            if callback:
                stats = {
                    'generation': generation,
                    'best_fitness': best_fitness,
                    'average_fitness': avg_fitness,
                    'diversity': diversity,
                    'mutation_rate': self.current_mutation_rate
                }
                callback(generation, stats)
            
            # Create new generation
            parents = self.selection(self.population)
            new_population = []
            
            for i in range(0, len(parents), 2):
                if i + 1 < len(parents):
                    child1, child2 = self.crossover(parents[i], parents[i + 1])
                    new_population.extend([
                        self.mutate(child1),
                        self.mutate(child2)
                    ])
                else:
                    new_population.append(self.mutate(parents[i]))
            
            # Apply elitism
            self.population = self.apply_elitism(self.population, new_population)
            
            # Adapt parameters
            self.adapt_parameters()
            
            if generation % 100 == 0 or generation < 10:
                print(f"Generation {generation:4d}: Best={best_fitness:.6f}, "
                      f"Avg={avg_fitness:.6f}, Div={diversity:.4f}")
        
        # Return best individual
        best_individual = max(self.population, key=lambda ind: ind.fitness)
        print(f"🏆 Evolution complete! Best fitness: {best_individual.fitness:.6f}")
        
        return best_individual
    
    def stop(self):
        """Stop the evolutionary process."""
        self.is_running = False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        if not self.best_fitness_history:
            return {}
        
        return {
            'generations_run': len(self.best_fitness_history),
            'best_fitness': max(self.best_fitness_history),
            'final_average_fitness': self.average_fitness_history[-1] if self.average_fitness_history else 0,
            'convergence_rate': np.mean(np.diff(self.best_fitness_history)),
            'population_diversity': self.diversity_history[-1] if self.diversity_history else 0,
            'fitness_history': self.best_fitness_history,
            'diversity_history': self.diversity_history
        }
    
    def save_state(self, filepath: str):
        """Save optimizer state to file."""
        state = {
            'generation': self.generation,
            'population': [
                {
                    'genome': ind.genome,
                    'fitness': ind.fitness,
                    'age': ind.age,
                    'generation': ind.generation,
                    'performance_history': ind.performance_history
                }
                for ind in self.population
            ],
            'parameters': {
                'population_size': self.parameters.population_size,
                'mutation_rate': self.parameters.mutation_rate,
                'crossover_rate': self.parameters.crossover_rate,
                'elitism_ratio': self.parameters.elitism_ratio,
                'tournament_size': self.parameters.tournament_size,
                'max_generations': self.parameters.max_generations
            },
            'statistics': self.get_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str):
        """Load optimizer state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.generation = state['generation']
        
        self.population = []
        for ind_data in state['population']:
            individual = Individual(
                genome=ind_data['genome'],
                fitness=ind_data['fitness'],
                age=ind_data['age'],
                generation=ind_data['generation'],
                performance_history=ind_data['performance_history']
            )
            self.population.append(individual)


class EvolutionaryStrategy:
    """Evolutionary Strategy (ES) optimizer for continuous optimization."""
    
    def __init__(self, 
                 fitness_function: FitnessFunction,
                 mu: int = 15,  # Number of parents
                 lambda_: int = 100,  # Number of offspring
                 sigma: float = 0.1):  # Initial step size
        self.fitness_function = fitness_function
        self.mu = mu
        self.lambda_ = lambda_
        self.sigma = sigma
        
        self.generation = 0
        self.best_fitness_history = []
        
    def evolve(self, 
               initial_solution: np.ndarray,
               max_generations: int = 1000,
               target_fitness: float = None) -> Tuple[np.ndarray, float]:
        """Run (μ, λ)-ES evolution."""
        
        # Initialize population
        dimension = len(initial_solution)
        population = []
        
        for _ in range(self.mu):
            individual = initial_solution + np.random.normal(0, self.sigma, dimension)
            individual = np.clip(individual, 0.0, 1.0)  # Ensure bounds
            population.append(individual)
        
        print(f"🧬 Starting ES optimization with μ={self.mu}, λ={self.lambda_}")
        
        for generation in range(max_generations):
            self.generation = generation
            
            # Evaluate population
            fitnesses = []
            for individual in population:
                genome_individual = Individual(genome=individual.tolist())
                fitness = self.fitness_function.evaluate(genome_individual, {})
                fitnesses.append((individual, fitness))
            
            # Sort by fitness
            fitnesses.sort(key=lambda x: x[1], reverse=True)
            best_fitness = fitnesses[0][1]
            self.best_fitness_history.append(best_fitness)
            
            # Check convergence
            if target_fitness and best_fitness >= target_fitness:
                print(f"🎯 ES converged at generation {generation}")
                return fitnesses[0][0], fitnesses[0][1]
            
            # Select parents (top μ individuals)
            parents = [ind for ind, fit in fitnesses[:self.mu]]
            
            # Generate λ offspring
            offspring = []
            for _ in range(self.lambda_):
                # Select random parent
                parent = parents[np.random.randint(len(parents))]
                
                # Mutate
                child = parent + np.random.normal(0, self.sigma, dimension)
                child = np.clip(child, 0.0, 1.0)
                offspring.append(child)
            
            # Next population is the offspring (μ, λ-ES)
            population = offspring
            
            # Adapt step size (simple 1/5 rule)
            if generation > 0 and generation % 10 == 0:
                recent_improvement = (self.best_fitness_history[-1] - 
                                    self.best_fitness_history[-10]) / 10
                if recent_improvement > 0.01:
                    self.sigma *= 1.2  # Increase step size
                else:
                    self.sigma *= 0.8  # Decrease step size
                
                self.sigma = max(0.001, min(0.5, self.sigma))  # Keep reasonable bounds
            
            if generation % 100 == 0:
                print(f"ES Generation {generation}: Best={best_fitness:.6f}, σ={self.sigma:.4f}")
        
        # Return best solution
        final_fitnesses = []
        for individual in population:
            genome_individual = Individual(genome=individual.tolist())
            fitness = self.fitness_function.evaluate(genome_individual, {})
            final_fitnesses.append((individual, fitness))
        
        best_solution, best_fitness = max(final_fitnesses, key=lambda x: x[1])
        print(f"🏆 ES complete! Best fitness: {best_fitness:.6f}")
        
        return best_solution, best_fitness


# Example usage and testing
def create_test_fitness_function():
    """Create a test fitness function for demonstration."""
    
    # Generate synthetic test data
    test_data = []
    for i in range(10):
        samples = [(f"input_{j}", f"target_{j%3}") for j in range(20)]
        test_samples = [(f"test_{j}", f"target_{j%3}") for j in range(10)]
        test_data.append({
            'samples': samples,
            'test_samples': test_samples
        })
    
    return HDCAlgorithmFitness(test_data, ['accuracy', 'latency'])


if __name__ == "__main__":
    # Demonstration of genetic optimizer
    fitness_func = create_test_fitness_function()
    
    params = EvolutionaryParameters(
        population_size=50,
        max_generations=200,
        mutation_rate=0.1,
        crossover_rate=0.8
    )
    
    optimizer = GeneticOptimizer(fitness_func, params, seed=42)
    
    def progress_callback(generation: int, stats: Dict[str, Any]):
        if generation % 50 == 0:
            print(f"Progress: Gen {generation}, Best: {stats['best_fitness']:.4f}")
    
    # Run optimization
    best_individual = optimizer.evolve(
        genome_length=6,  # HDC parameters
        callback=progress_callback
    )
    
    print("Best HDC parameters found:")
    fitness_func_instance = HDCAlgorithmFitness([], [])
    params = fitness_func_instance._decode_genome(best_individual.genome)
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Save results
    optimizer.save_state("genetic_optimizer_results.json")
    print("Results saved to genetic_optimizer_results.json")