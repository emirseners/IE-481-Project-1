from gurobipy import Model, GRB, quicksum, tuplelist
import numpy as np
import matplotlib.pyplot as plt
import time
import networkx as nx
import pandas as pd

def Metric_Complete_Graph(n, seed=None, x_range=(0, 100), y_range=(0, 100)):
    np.random.seed(seed)
    x = np.random.uniform(*x_range, n)
    y = np.random.uniform(*y_range, n)
    coordinates = np.column_stack((x, y))
    distance_matrix = np.sqrt((x[:, None] - x[None, :])**2 + (y[:, None] - y[None, :])**2)
    return coordinates, distance_matrix

def Draw_Graph(coordinates, visiting_order=None):
    plt.figure(figsize=(8, 8))
    plt.scatter(coordinates[:, 0], coordinates[:, 1], color='blue', zorder=5)

    for idx, (x, y) in enumerate(coordinates):
        plt.text(x, y, str(idx), fontsize=12, ha='right', va='bottom', color='black')

    if visiting_order:
        ordered_coords = coordinates[visiting_order + [visiting_order[0]]]
        plt.plot(ordered_coords[:, 0], ordered_coords[:, 1], 'r-', zorder=1)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('TSP Visiting Order')
    plt.axis('equal')
    plt.xlim(-10, 110)
    plt.ylim(-10, 110)
    plt.show()

def Nearest_Neighbor_Algorithm(graph, start_node=0):
    distance_matrix = np.array(graph[1])
    number_of_nodes = distance_matrix.shape[0]
    start_time = time.time()

    visiting_order = [start_node]
    total_distance = 0
    current_node = start_node
    unvisited = set(range(number_of_nodes))
    unvisited.remove(start_node)

    while unvisited:
        next_node = min(unvisited, key=lambda node: distance_matrix[current_node, node])
        total_distance += distance_matrix[current_node, next_node]
        visiting_order.append(next_node)
        current_node = next_node
        unvisited.remove(next_node)

    total_distance += distance_matrix[current_node, start_node]
    visiting_order.append(start_node)
    algorithm_time = time.time() - start_time

    return visiting_order, total_distance, algorithm_time

def Christofides_Algorithm(graph):
    coordinates, distance_matrix = graph
    n = len(coordinates)

    mst_complete_graph = nx.Graph()
    for i in range(n-1):
        for j in range(i+1, n):
            mst_complete_graph.add_edge(i, j, weight=distance_matrix[i, j])

    start_time = time.time()
    minimum_spanning_tree = nx.minimum_spanning_tree(mst_complete_graph)
    odd_degree_nodes = [node for node, degree in minimum_spanning_tree.degree() if degree % 2 == 1]

    minc_perfect_matching_input = nx.Graph()
    for i in range(len(odd_degree_nodes) - 1):
        for j in range(i + 1, len(odd_degree_nodes)):
            minc_perfect_matching_input.add_edge(odd_degree_nodes[i], odd_degree_nodes[j], weight=distance_matrix[odd_degree_nodes[i], odd_degree_nodes[j]])
    minc_perfect_matching = nx.min_weight_matching(minc_perfect_matching_input, weight='weight')

    graph_with_eulerian_circuit = nx.MultiGraph(minimum_spanning_tree)
    graph_with_eulerian_circuit.add_edges_from(minc_perfect_matching)
    eulerian_circuit = list(nx.eulerian_circuit(graph_with_eulerian_circuit))

    visited = set()
    hamiltonian_tour = []
    for u, _ in eulerian_circuit:
        if u not in visited:
            hamiltonian_tour.append(u)
            visited.add(u)

    hamiltonian_tour.append(hamiltonian_tour[0])

    total_distance = sum(distance_matrix[hamiltonian_tour[i], hamiltonian_tour[i + 1]] for i in range(len(hamiltonian_tour) - 1))

    end_time = time.time()
    algorithm_time = end_time - start_time

    return hamiltonian_tour, total_distance, algorithm_time

def Find_Subtour(edges, n):
    graph = nx.DiGraph(edges)
    cycles = list(nx.simple_cycles(graph))
    if cycles:
        return min(cycles, key=len)
    else:
        return list(range(n))

def Subtour_Elimination(model, where):
    if where == GRB.Callback.MIPSOL:
        values = model.cbGetSolution(model._x_ij)
        edges = [(i, j) for i, j in model._x_ij.keys() if values[i, j] > 1e-5]
        subtour = Find_Subtour(edges, model._n)
        if len(subtour) < model._n:
            model.cbLazy(quicksum(model._x_ij[i, j] for i in subtour for j in subtour if i != j) <= len(subtour) - 1)

def Tsp_Ip(graph, time_limit=3600, mip_gap=0.0001):
    coordinates, distance_matrix = graph
    n = len(coordinates)

    model = Model('Tsp_Ip')
    model.setParam('OutputFlag', True)
    model.setParam('MIPGap', mip_gap)
    model.setParam('TimeLimit', time_limit)
    model.setParam('LazyConstraints', 1)
    #model.setParam('Cuts', 3)

    x_ij = model.addVars([(i, j) for i in range(n) for j in range(n) if i != j], vtype=GRB.BINARY, name="x")

    model.setObjective(quicksum(distance_matrix[i][j] * x_ij[i, j] for i in range(n) for j in range(n) if i != j), GRB.MINIMIZE)

    model.addConstrs(quicksum(x_ij[i, j] for j in range(n) if i != j) == 1 for i in range(n))
    model.addConstrs(quicksum(x_ij[i, j] for i in range(n) if i != j) == 1 for j in range(n))

    model._x_ij = x_ij
    model._n = n

    model.optimize(Subtour_Elimination)
    
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        vals = model.getAttr(GRB.Attr.X, x_ij)
        selected = tuplelist((i, j) for i, j in x_ij.keys() if vals[i, j] > 1e-5)
        tour_dict = {i: j for i, j in selected}
        start = selected[0][0]
        optimal_tour = [start]

        while True:
            next_node = tour_dict[optimal_tour[-1]]
            if next_node == start:
                break
            optimal_tour.append(next_node)

        optimal_tour.append(start)

        print(f'Optimal tour: {optimal_tour}')
        print(f'Optimal cost: {model.objVal}')
        print(f'Optimization Time: {model.Runtime:.2f} seconds')
    else:
        print("Infeasible")

    return optimal_tour, model.objVal, model.Runtime

def Max_N_in_Time(algo, time_limit=600, initial_n=300, step=2, trials=5):
    n = initial_n
    while True:
        exceeded_count = 0
        for _ in range(trials):
            graph = Metric_Complete_Graph(n)
            if algo == 'NN':
                _, _, algorithm_time = Nearest_Neighbor_Algorithm(graph)
            elif algo == 'Christofides':
                _, _, algorithm_time = Christofides_Algorithm(graph)
            elif algo == 'TSP IP':
                _, _, algorithm_time = Tsp_Ip(graph)
            if algorithm_time >= time_limit:
                exceeded_count += 1
        if exceeded_count == trials:
            return n - step
        else:
            n += step

number_of_experiments = 10
number_of_vertices = [10, 20, 50, 100, 200]
algorithms = ['NN', 'Christofides', 'TSP IP']

results = []

for n in number_of_vertices:
    for experiment in range(number_of_experiments):
        graph = Metric_Complete_Graph(n)
        for algo in algorithms:
            if algo == 'NN':
                visiting_order, total_distance, algorithm_time = Nearest_Neighbor_Algorithm(graph)
            elif algo == 'Christofides':
                visiting_order, total_distance, algorithm_time = Christofides_Algorithm(graph)
            elif algo == 'TSP IP':
                visiting_order, total_distance, algorithm_time = Tsp_Ip(graph)

            results.append({
                'Number_of_Vertices': n,
                'Experiment': experiment + 1,
                'Algorithm': algo,
                'Total_Distance': total_distance,
                'Algorithm_Time': algorithm_time
            })

results_df = pd.DataFrame(results)

summary_df = results_df.groupby(['Number_of_Vertices', 'Algorithm']).agg(
    Average_Total_Distance=('Total_Distance', 'mean'),
    Worst_Total_Distance=('Total_Distance', 'max'),
    Average_Runtime=('Algorithm_Time', 'mean'),
    Max_Runtime=('Algorithm_Time', 'max')
).reset_index()

largest_solvable = {algo: Max_N_in_Time(algo, time_limit=600) for algo in algorithms}
largest_solvable_df = pd.DataFrame([{'Algorithm': algo, 'Largest_Solvable_n_within_10min': n} for algo, n in largest_solvable.items()])

with pd.ExcelWriter('tsp_algorithms_results.xlsx') as writer:
    results_df.to_excel(writer, sheet_name='All_Experiments', index=False)
    summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
    largest_solvable_df.to_excel(writer, sheet_name='Largest_Solvable_10min', index=False)