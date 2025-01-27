import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import sys
import time
from collections import deque

def read_graph(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    n = int(lines[0].strip())
    weights = list(map(int, lines[1].strip().split()))
    matrix = np.array([list(map(int, line.strip().split())) for line in lines[2:]])
    
    return n, weights, matrix

def initial_solution(n, matrix):
    cover = set()
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] == 1:
                cover.add(i)
                cover.add(j)
    return list(cover)

def evaluate_solution(cover, weights):
    return sum(weights[v] for v in cover)

def get_neighbors(solution, n, matrix):
    neighbors = []
    solution_set = set(solution)

    for v in solution:
        new_solution = solution_set.copy()
        new_solution.remove(v)
        if is_valid_cover(list(new_solution), matrix, n):
            neighbors.append(list(new_solution))

    not_in_solution = set(range(n)) - solution_set
    for v in not_in_solution:
        new_solution = solution_set.copy()
        new_solution.add(v)
        if is_valid_cover(list(new_solution), matrix, n):
            neighbors.append(list(new_solution))
    
    return neighbors

def tabu_search(n, weights, matrix, max_iter=200, tabu_tenure=8, max_no_improvement=5):
    current_solution = initial_solution(n, matrix)
    best_solution = current_solution
    best_cost = evaluate_solution(best_solution, weights)
    tabu_list = deque(maxlen=tabu_tenure)
    no_improvement_count = 0
    visited_solutions = set()

    for iteration in range(max_iter):
        neighbors = get_neighbors(current_solution, n, matrix)
        if not neighbors:
            break
        best_neighbor = None
        best_neighbor_cost = float('inf')

        for neighbor in neighbors:
            neighbor_tuple = tuple(sorted(neighbor))
            if neighbor_tuple not in tabu_list and neighbor_tuple not in visited_solutions:
                cost = evaluate_solution(neighbor, weights)
                if cost < best_neighbor_cost:
                    best_neighbor = neighbor
                    best_neighbor_cost = cost

        if best_neighbor is not None:
            current_solution = best_neighbor
            neighbor_tuple = tuple(sorted(best_neighbor))
            tabu_list.append(neighbor_tuple)
            visited_solutions.add(neighbor_tuple)
            
            if best_neighbor_cost < best_cost:
                best_solution = best_neighbor
                best_cost = best_neighbor_cost
                no_improvement_count = 0
            else:
                no_improvement_count += 1
        else:
            no_improvement_count += 1

        if no_improvement_count >= max_no_improvement:
            break
    
    return best_solution, best_cost, iteration + 1

def is_valid_cover(solution, matrix, n):
    covered_edges = set()
    for v in solution:
        for u in range(n):
            if matrix[v][u] == 1:
                covered_edges.add((min(v, u), max(v, u)))
    return len(covered_edges) == np.sum(matrix) // 2

def plot_graph(matrix, solution):
    G = nx.Graph()
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            if matrix[i][j] == 1:
                G.add_edge(i, j)
    
    pos = nx.spring_layout(G)
    node_colors = ['red' if node in solution else 'blue' for node in G.nodes()]
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color='gray')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 VertexCoverSlow.py <input_file>")
        sys.exit(1)
    
    filename = sys.argv[1]
    n, weights, matrix = read_graph(filename)
    start_time = time.time()
    best_solution, best_cost, iteration_count = tabu_search(n, weights, matrix)
    end_time = time.time()
    
    print("Best Solution:", best_solution)
    print("Best Cost:", best_cost)
    print("Total Iterations:", iteration_count)
    print(f"Algorithm Execution Time: {end_time - start_time:.4f} seconds")
    
    plot_graph(matrix, best_solution)
