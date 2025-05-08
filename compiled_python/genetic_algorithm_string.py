import random
import string
# --- GENERATED CODE ---
def create_chromosome(length: int) -> str:
    """Creates a new chromosome (string) of a given length consisting of random printable ASCII characters."""
    return ''.join(random.choice(string.printable) for _ in range(length))
def calculate_fitness(chromosome: str, target: str) -> int:
    if len(chromosome) != len(target):
        raise ValueError("Chromosome and target must be of the same length.")
    fitness_score = sum(1 for c, t in zip(chromosome, target) if c == t)
    return fitness_score
def select_parents(population, fitness_map, num_parents_to_select):
    # Sort the population by fitness in descending order
    sorted_population = sorted(population, key=lambda chromosome: fitness_map[chromosome], reverse=True)
    # Select the top individuals based on the specified number of parents to select
    selected_parents = sorted_population[:num_parents_to_select]
    return selected_parents
def crossover(parent1: str, parent2: str) -> str:
    if len(parent1) != len(parent2):
        raise ValueError('Parent chromosomes must be of the same length.')
    crossover_point = random.randint(1, len(parent1) - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child
def mutate(chromosome: str, mutation_rate: float) -> str:
    mutated_chromosome = list(chromosome)
    for i in range(len(mutated_chromosome)):
        if random.random() < mutation_rate:
            mutated_chromosome[i] = random.choice(string.printable.strip())
    return ''.join(mutated_chromosome)
def create_next_generation(current_population, target_string, mutation_rate, elitism_count, population_size):
    # Step 1: Calculate Fitness
    fitness_scores = {chromosome: calculate_fitness(chromosome, target_string) for chromosome in current_population}

    # Step 2: Elitism
    sorted_population = sorted(current_population, key=lambda chrom: fitness_scores[chrom], reverse=True)
    elites = sorted_population[:elitism_count]

    # Step 3: Parent Selection
    parents = select_parents(current_population, fitness_scores, population_size - elitism_count)

    # Step 4: Crossover & Mutation
    offspring = []
    while len(offspring) < (population_size - elitism_count):
        parent1, parent2 = random.sample(parents, 2)
        child = crossover(parent1, parent2)
        mutated_child = mutate(child, mutation_rate)
        offspring.append(mutated_child)

    # Step 5: Form New Population
    new_population = elites + offspring
    return new_population
def initialize_population(population_size: int, chromosome_length: int):
    """Creates an initial population of random chromosomes."""
    def create_chromosome(length):
        return ''.join(random.choices(string.printable[:-6], k=length))  # Exclude non-printable characters

    population = [create_chromosome(chromosome_length) for _ in range(population_size)]
    return population
def run_evolutionary_process(target_string, chromosome_length, population_size, mutation_rate, elitism_count, num_generations):
    # Initialize Population
    current_population = initialize_population(population_size, chromosome_length)

    # Initialize Tracking
    best_overall_fitness = -1
    best_overall_chromosome = None

    # Loop through Generations
    for generation in range(num_generations):
        # Calculate Fitness Map
        fitness_map = {chromosome: calculate_fitness(chromosome, target_string) for chromosome in current_population}
        
        # Find Generation Best
        current_best_chromosome = max(fitness_map, key=fitness_map.get)
        current_best_fitness = fitness_map[current_best_chromosome]

        # Update Overall Best
        if current_best_fitness > best_overall_fitness:
            best_overall_fitness = current_best_fitness
            best_overall_chromosome = current_best_chromosome
            print(f"Gen {generation}: Best Fitness: {best_overall_fitness}/{chromosome_length} -> {best_overall_chromosome}")

        # Check for Solution
        if best_overall_fitness == chromosome_length:
            print("Solution found!")
            return best_overall_chromosome

        # Evolve
        current_population = create_next_generation(current_population, target_string, mutation_rate, elitism_count, population_size)

    # After Loop
    print("Finished generations. Best found:")
    print(best_overall_chromosome, best_overall_fitness)
    return best_overall_chromosome

# --- UNCOMPILED CODE ---
if __name__ == '__main__':

    def run_ga_experiment():
        TARGET_STRING = "AuraIsCool!"
        CHROMOSOME_LENGTH = len(TARGET_STRING)
        POPULATION_SIZE = 100
        MUTATION_RATE = 0.02  # 2% chance per character
        ELITISM_COUNT = 5     # Carry over the best 5 individuals
        NUMBER_OF_GENERATIONS = 300 # Reduced for potentially faster demonstration

        print(f"Target: '{TARGET_STRING}' (Length: {CHROMOSOME_LENGTH})")
        print(f"Population Size: {POPULATION_SIZE}, Mutation Rate: {MUTATION_RATE*100}%, Elitism: {ELITISM_COUNT}, Generations: {NUMBER_OF_GENERATIONS}")

        final_best_chromosome = run_evolutionary_process(
            target_string=TARGET_STRING,
            chromosome_length=CHROMOSOME_LENGTH,
            population_size=POPULATION_SIZE,
            mutation_rate=MUTATION_RATE,
            elitism_count=ELITISM_COUNT,
            num_generations=NUMBER_OF_GENERATIONS
        )

        print(f"\\nEvolutionary process complete. Final best chromosome from run: '{final_best_chromosome}'")

    run_ga_experiment()