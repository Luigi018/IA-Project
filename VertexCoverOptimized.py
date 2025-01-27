import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import heapq
import sys
import time
from collections import deque

def read_graph(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    n = int(lines[0].strip())
    weights = np.array(list(map(int, lines[1].strip().split())))
    matrix = np.array([list(map(int, line.strip().split())) for line in lines[2:]], dtype=np.uint8)
    
    return n, weights, matrix

def initial_solution(n, matrix):
    cover = set()
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i, j]:
                cover.add(i)
                cover.add(j)
    return list(cover)

def evaluate_solution(cover, weights):
    return np.sum(weights[list(cover)])

def is_valid_cover(solution, matrix, n):
    covered = np.zeros((n, n), dtype=np.uint8)
    for v in solution:
        covered[v, :] |= matrix[v, :]
        covered[:, v] |= matrix[:, v]
    return np.all(covered | (matrix == 0))

def get_neighbors_optimized(solution, n, matrix):
    neighbors = []
    solution_set = set(solution)
    not_in_solution = set(range(n)) - solution_set
    
    for v in solution:
        new_solution = solution_set - {v}
        if is_valid_cover(new_solution, matrix, n):
            neighbors.append(new_solution)
    
    for v in not_in_solution:
        new_solution = solution_set | {v}
        if is_valid_cover(new_solution, matrix, n):
            neighbors.append(new_solution)
    
    return neighbors

def tabu_search_optimized(n, weights, matrix, max_iter=200, tabu_tenure=8, max_no_improvement=5):
    start_time = time.time()
    
    current_solution = set(initial_solution(n, matrix))
    best_solution = current_solution.copy()
    best_cost = evaluate_solution(best_solution, weights)
    tabu_list = deque(maxlen=tabu_tenure)
    visited_solutions = set()
    no_improvement_count = 0
    iteration_count = 0 

    for iteration in range(max_iter):
        iteration_count += 1
        neighbors = get_neighbors_optimized(current_solution, n, matrix)
        if not neighbors:
            break

        heap = []
        for neighbor in neighbors:
            neighbor_tuple = frozenset(neighbor)
            if neighbor_tuple not in tabu_list and neighbor_tuple not in visited_solutions:
                cost = evaluate_solution(neighbor, weights)
                heapq.heappush(heap, (cost, neighbor))

        if not heap:
            break

        best_neighbor_cost, best_neighbor = heapq.heappop(heap)
        current_solution = best_neighbor
        tabu_list.append(frozenset(best_neighbor))
        visited_solutions.add(frozenset(best_neighbor))
        
        if best_neighbor_cost < best_cost:
            best_solution = best_neighbor
            best_cost = best_neighbor_cost
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        if no_improvement_count >= max_no_improvement:
            break

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return best_solution, best_cost, iteration_count, elapsed_time

def plot_graph(matrix, solution):
    G = nx.Graph()
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i, j]:
                G.add_edge(i, j)
    
    pos = nx.spring_layout(G)
    node_colors = ['red' if node in solution else 'blue' for node in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color='gray')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 VertexCoverOptimized.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    n, weights, matrix = read_graph(input_file)
    best_solution, best_cost, iteration_count, elapsed_time = tabu_search_optimized(n, weights, matrix)
    
    print("Best Solution:", best_solution)
    print("Best Cost:", best_cost)
    print("Total Iterations:", iteration_count)
    print(f"Algorithm Execution Time: {elapsed_time:.4f} seconds")
    
    plot_graph(matrix, best_solution)
