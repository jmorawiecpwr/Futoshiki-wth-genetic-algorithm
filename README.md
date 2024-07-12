# Futoshiki Genetic Algorithm Solver

This project implements a genetic algorithm to solve the Futoshiki puzzle. The algorithm evolves a population of potential solutions over several generations to find the best solution that satisfies the constraints of the puzzle. The project also measures and visualizes the execution time of the algorithm over multiple trials.

## Features

- **Genetic Algorithm Implementation:** Uses selection, crossover, and mutation operations to evolve solutions.
- **Fitness Evaluation:** Scores solutions based on the satisfaction of Futoshiki constraints.
- **Performance Analysis:** Measures the execution time of the algorithm over 100 trials and visualizes the results.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/your-username/futoshiki-genetic-algorithm.git
    ```
2. Navigate to the project directory:
    ```sh
    cd futoshiki-genetic-algorithm
    ```
3. Install the required dependencies:
    ```sh
    pip install numpy matplotlib
    ```

## Usage

1. Run the main script to execute the genetic algorithm:
    ```sh
    python main.py
    ```
2. The script will output the execution times for each trial and display a plot of the execution times.

## Code Explanation

- `generate_board(n)`: Generates a random Futoshiki board of size `n x n`.
- `fitness(board)`: Evaluates the fitness of a board based on row, column, and constraint satisfaction.
- `selection(population, fitnesses)`: Selects the two best boards from the population.
- `crossover(parent1, parent2)`: Creates two children boards by crossing over two parent boards.
- `mutate(board)`: Mutates a board by swapping two elements in a randomly selected row.
- `genetic_algorithm()`: Main function that executes the genetic algorithm over a specified number of generations and returns the best board and its fitness.
- Performance measurement and visualization of the execution time over 100 trials.

## Example

```python
import numpy as np
import random
import time
import matplotlib.pyplot as plt

# Configuration
n = 9  # Size of the Futoshiki board
pop_size = 100  # Population size
num_generations = 1000  # Number of generations
mutation_rate = 0.1  # Mutation rate

# Constraints for the Futoshiki puzzle
constraints = [
    (0, 0, 1, 0),
    (2, 1, 2, 2),
    (3, 3, 3, 4),
    (4, 2, 3, 2),
    (1, 4, 0, 4),
    (5, 5, 5, 6),
    (7, 8, 6, 8),
    (8, 0, 8, 1),
    (4, 4, 5, 4),
    (2, 6, 2, 7)
]

# Function implementations (generate_board, fitness, selection, crossover, mutate, genetic_algorithm)

# Performance measurement and visualization
execution_times = []
for i in range(100):
    start_time = time.time()
    best_board, best_fitness = genetic_algorithm()
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times.append(execution_time)
    print(f"Trial {i+1}: Execution Time = {execution_time:.4f} s")

min_time = min(execution_times)
max_time = max(execution_times)
avg_time = sum(execution_times) / len(execution_times)

print(f"Shortest Execution Time: {min_time:.4f} s")
print(f"Longest Execution Time: {max_time:.4f} s")
print(f"Average Execution Time: {avg_time:.4f} s")

plt.plot(execution_times, marker='o')
plt.xlabel('Trial Number')
plt.ylabel('Execution Time (s)')
plt.title('Execution Time of Genetic Algorithm over Trials')
plt.show()
