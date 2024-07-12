import numpy as np
import random
import time
import matplotlib.pyplot as plt

n = 9  # rozmiar planszy Futoshiki
pop_size = 100  # wielkość populacji
num_generations = 1000  # liczba generacji
mutation_rate = 0.1  # współczynnik mutacji

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

def generate_board(n):
    board = np.zeros((n, n), dtype=int)
    for i in range(n):
        board[i, :] = np.random.permutation(range(1, n + 1))
    return board

def fitness(board):
    score = 0

    for i in range(n):
        if len(set(board[i, :])) == n:
            score += 1
        if len(set(board[:, i])) == n:
            score += 1

    for (x1, y1, x2, y2) in constraints:
        if x1 == x2 and y1 == y2:
            continue
        if board[x1, y1] > board[x2, y2]:
            score += 1

    return score

def selection(population, fitnesses):
    indices = np.argsort(fitnesses)[-2:]  # wybór dwóch najlepszych
    return [population[idx] for idx in indices]

def crossover(parent1, parent2):
    point = np.random.randint(1, n - 1)
    child1 = np.vstack((parent1[:point, :], parent2[point:, :]))
    child2 = np.vstack((parent2[:point, :], parent1[point:, :]))
    return child1, child2

def mutate(board):
    if np.random.rand() < mutation_rate:
        row = np.random.randint(0, n)
        col1, col2 = np.random.choice(n, 2, replace=False)
        board[row, col1], board[row, col2] = board[row, col2], board[row, col1]
    return board

def genetic_algorithm():
    population = [generate_board(n) for _ in range(pop_size)]
    best_board = None
    best_fitness = 0

    for gen in range(num_generations):
        fitnesses = [fitness(board) for board in population]
        best_idx = np.argmax(fitnesses)

        if fitnesses[best_idx] > best_fitness:
            best_fitness = fitnesses[best_idx]
            best_board = population[best_idx]

        if best_fitness == 2 * n + len(constraints):
            break

        selected = selection(population, fitnesses)
        population = []

        while len(population) < pop_size:
            parent1, parent2 = selected
            child1, child2 = crossover(parent1, parent2)
            population.append(mutate(child1))
            population.append(mutate(child2))

    return best_board, best_fitness

execution_times = []
for i in range(100):
    start_time = time.time()
    best_board, best_fitness = genetic_algorithm()
    end_time = time.time()
    execution_time = end_time - start_time
    execution_times.append(execution_time)
    print(f"Próba {i+1}: Czas wykonania = {execution_time:.4f} s")

min_time = min(execution_times)
max_time = max(execution_times)
avg_time = sum(execution_times) / len(execution_times)

print(f"Najkrótszy czas wykonania: {min_time:.4f} s")
print(f"Najdłuższy czas wykonania: {max_time:.4f} s")
print(f"Średni czas wykonania: {avg_time:.4f} s")

plt.plot(execution_times, marker='o')
plt.xlabel('Numer próby')
plt.ylabel('Czas wykonania (s)')
plt.title('Czas wykonania algorytmu genetycznego w kolejnych próbach')
plt.show()
