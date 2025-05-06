# Standard Python Imports
import random
import string

# Built-in imports (added by compiler)
import os
import sys
import re
from typing import List, Dict, Any, Optional, TypeAlias

# Type Aliases from 'desc' blocks (for clarity)
target_string: TypeAlias = Any  # Hello, World!
population_size: TypeAlias = Any  # 100
mutation_rate: TypeAlias = Any  # 0.01
generations: TypeAlias = Any  # 500
charset: TypeAlias = int  # Use string.printable

# --- Aura Block: aura_function:create_individual ---


def create_individual() -> str:
    individual = ''.join(random.choice(string.printable)
                         for _ in range(len('Hello, World!')))
    return individual

# --- Aura Block: aura_function:calculate_fitness ---


def calculate_fitness(individual: str) -> float:
    """Calculates fitness as the proportion of characters matching target_string."""
    target_string = 'Hello, World!'
    score = sum(1 for a, b in zip(individual, target_string) if a == b)
    fitness = score / len(target_string)
    return fitness

# --- Aura Block: aura_function:select_parents ---


def select_parents(population: list, fitnesses: list) -> tuple:
    total_fitness = sum(fitnesses)
    probabilities = [fitness / total_fitness for fitness in fitnesses]
    parent1_index = random.choices(
        range(len(population)), weights=probabilities)[0]
    parent2_index = random.choices(
        range(len(population)), weights=probabilities)[0]
    while parent2_index == parent1_index:
        parent2_index = random.choices(
            range(len(population)), weights=probabilities)[0]
    return (population[parent1_index], population[parent2_index])

# --- Aura Block: aura_function:crossover ---


def crossover(parent1: str, parent2: str) -> str:
    crossover_point = random.randint(0, len(parent1))
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child

# --- Aura Block: aura_function:mutate ---


def mutate(individual: str) -> str:
    """Mutates characters in the individual based on mutation_rate using charset."""
    mutated_individual = ''
    for char in individual:
        if random.random() < 0.01:
            mutated_individual += random.choice(string.printable)
        else:
            mutated_individual += char
    return mutated_individual

# --- Aura Block: aura_function:run_evolution ---


def run_evolution():
    population = [create_individual() for _ in range(100)]
    for generation in range(500):
        fitnesses = [calculate_fitness(individual)
                     for individual in population]
        best_fitness = max(fitnesses)
        best_individual = population[fitnesses.index(best_fitness)]
        print(generation, best_fitness, best_individual)
        if best_individual == 'Hello, World!':
            break
        next_generation = []
        for _ in range(100):
            p1, p2 = select_parents(population, fitnesses)
            child = crossover(p1, p2)
            child = mutate(child)
            next_generation.append(child)
        population = next_generation
    print(best_individual)


# --- Main Block ---
if __name__ == "__main__":
    run_evolution()
