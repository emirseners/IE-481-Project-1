from gurobipy import Model, GRB, quicksum, tuplelist
import numpy as np
import matplotlib.pyplot as plt
import time
import networkx as nx
import pandas as pd

def Metric_Complete_Graph(n, seed=None, x_range=(0, 100), y_range=(0, 100)): #Function to generate a complete metric graph, with n vertices
    np.random.seed(seed)                                                     #If random seed is defined for experiments, it is setted
    x = np.random.uniform(*x_range, n)                                       #Generating n many uniformly distributed points in the x-axis between x_range
    y = np.random.uniform(*y_range, n)                                       #Generating n many uniformly distributed points in the y-axis between y_range
    coordinates = np.column_stack((x, y))                                    #Combining x and y coordinates into a single array
    distance_matrix = np.sqrt((x[:, None] - x[None, :])**2 + (y[:, None] - y[None, :])**2)    #Defining a square matrix which consists of the Euclidean distances between coordinates
    return coordinates, distance_matrix                                      #Returning the coordinates and the distance matrix, vertices and edge waits of the graph

def Draw_Graph(coordinates, visiting_order=None):                            #Function to draw the graph for visualization and checking the results
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

def Nearest_Neighbor_Algorithm(graph, start_node=0):  #Function to implement the Nearest Neighbor Algorithm, taking graph as input
    distance_matrix = np.array(graph[1])
    number_of_nodes = distance_matrix.shape[0]
    start_time = time.time()

    visiting_order = [start_node]           #Start node is determined to be arbitrarily in NN. Since graph is constructed randomly, it is not important which node is selected as start node. Thus node 0 for each time is also random selection
    total_distance = 0                      #Variable to track objective
    current_node = start_node
    unvisited = set(range(number_of_nodes)) #Set of unvisited nodes is created, which includes all nodes in the graph
    unvisited.remove(start_node)

    while unvisited:                                                                     #While there are unvisited nodes, the algorithm continues to work O(n)
        next_node = min(unvisited, key=lambda node: distance_matrix[current_node, node]) #Next node is defined among unvisited nodes, which has the minimum distance to the current node. This is conducted through linear scanning of the distances. O(n)
        total_distance += distance_matrix[current_node, next_node] #Update total distance with new distance
        visiting_order.append(next_node)                           #Add closest distanced node to the visiting order
        current_node = next_node                                   #Set current node to the next node
        unvisited.remove(next_node)                                #Remove the next node from the unvisited set

    total_distance += distance_matrix[current_node, start_node]    #Add the distance from the last node back to the start node to the total distance and complete the tour by appending start node again to the visiting order
    visiting_order.append(start_node)
    algorithm_time = time.time() - start_time                      #Calculate the time taken for the algorithm to run

    return visiting_order, total_distance, algorithm_time          #Return the visiting order, total distance and algorithm time

def Christofides_Algorithm(graph):                                 #Function to implement the Nearest Neighbor Algorithm, taking graph as input
    coordinates, distance_matrix = graph
    n = len(coordinates)

    mst_complete_graph = nx.Graph()                                #Defne a Graph object to be used as input for the minimum spanning tree
    for i in range(n-1):                                           #Add weighted edges to the graph, where weights are the distances between the nodes
        for j in range(i+1, n):
            mst_complete_graph.add_edge(i, j, weight=distance_matrix[i, j])

    start_time = time.time()
    minimum_spanning_tree = nx.minimum_spanning_tree(mst_complete_graph)                               #MST of G is found. Default argument is Kruskal's algorithm, Therefore complexity O(E log E)
    odd_degree_nodes = [node for node, degree in minimum_spanning_tree.degree() if degree % 2 == 1]    #Find the odd degree nodes in the MST. O(V)

    minc_perfect_matching_input = nx.Graph()                                                           #Define a Graph object to be used as input for the min weight perfect matching.
    for i in range(len(odd_degree_nodes) - 1):                                                         #Define edges with weight between each pair of odd degree nodes. O(V^2)
        for j in range(i + 1, len(odd_degree_nodes)):
            minc_perfect_matching_input.add_edge(odd_degree_nodes[i], odd_degree_nodes[j], weight=distance_matrix[odd_degree_nodes[i], odd_degree_nodes[j]])
    minc_perfect_matching = nx.min_weight_matching(minc_perfect_matching_input, weight='weight')       #Find the minimum weight perfect matching of the odd degree nodes. O(V^3) for the blossom algorithm

    graph_with_eulerian_circuit = nx.MultiGraph(minimum_spanning_tree)        #Define a MultiGraph object to be used as input for the Eulerian circuit
    graph_with_eulerian_circuit.add_edges_from(minc_perfect_matching)         #Add the edges of the minimum weight perfect matching to the MST to create a graph where the degree of all vertices are even. O(E)
    eulerian_circuit = list(nx.eulerian_circuit(graph_with_eulerian_circuit)) #Find the Eulerian circuit of the graph. O(2V)

    visited = set()
    hamiltonian_tour = []
    for u, _ in eulerian_circuit:          #Iterate through the edges of the Eulerian circuit and add the nodes to the Hamiltonian tour if they have not been visited yet. O(2V)
        if u not in visited:               #If they have been visited, skip them. Since working on Metric Tsp instance, objective must be less than or equal to the circuit containing multiple visits.
            hamiltonian_tour.append(u)
            visited.add(u)

    hamiltonian_tour.append(hamiltonian_tour[0])      #Add the start node to the end of the Hamiltonian tour to complete the tour

    total_distance = sum(distance_matrix[hamiltonian_tour[i], hamiltonian_tour[i + 1]] for i in range(len(hamiltonian_tour) - 1))  #Calculate the total distance of the Hamiltonian tour.

    end_time = time.time()
    algorithm_time = end_time - start_time

    return hamiltonian_tour, total_distance, algorithm_time

def Find_Subtour(edges, n):      #Given the edges and n as input, the function returns the closed paths. The function uses NetworkX to create a directed graph and find the cycles in it. https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.cycles.simple_cycles.html
    graph = nx.DiGraph(edges)
    cycles = list(nx.simple_cycles(graph))
    if cycles:
        return min(cycles, key=len) #Return the shortest cycle found in the graph. If no cycles are found, return a list with length n.
    else:
        return list(range(n))

def Subtour_Elimination(model, where):  #Lazy constraint callback function to eliminate subtours in a TSP model. 
    if where == GRB.Callback.MIPSOL:                   #Check if new integer feasible solution is found, which would update the incumbent solution
        values = model.cbGetSolution(model._x_ij)      #Get the solution values of the decision variables
        edges = [(i, j) for i, j in model._x_ij.keys() if values[i, j] > 1e-5]
        subtour = Find_Subtour(edges, model._n)        #Find the subtour in the current solution using the Find_Subtour function
        if len(subtour) < model._n:                    #If the length of the subtour is less than n, it means that a subtour has been found, and a lazy constraint is added to eliminate it.
            model.cbLazy(quicksum(model._x_ij[i, j] for i in subtour for j in subtour if i != j) <= len(subtour) - 1)
            #Lazy constraint is added if the found feasible solution contains a subtour. Instead of defining all constraints at the beginning, lazy constraints are used to reduce the number of constraints in the model and speed up the optimization process.
            #Adding constraints for all cycles found is also an option, which is mode aggresive.

def Tsp_Ip(graph, time_limit=3600, mip_gap=0.0001):  #Function to implement the TSP IP algorithm, taking graph as input. Default MIP gap of Gurobi is used.
    coordinates, distance_matrix = graph
    n = len(coordinates)

    model = Model('Tsp_Ip')                      #We first define model
    model.setParam('OutputFlag', True)           #Set output flag to True to see the output of the model
    model.setParam('MIPGap', mip_gap)                
    model.setParam('TimeLimit', time_limit)    
    model.setParam('LazyConstraints', 1)         #Set lazy constraints to 1 to use lazy constraints for subtour elimination
    #model.setParam('Cuts', 3)                   #This parameter is tested which ensures more 'agressive cuts' but not used in presented experiments

    x_ij = model.addVars([(i, j) for i in range(n) for j in range(n) if i != j], vtype=GRB.BINARY, name="x")      #Decision varibles are defined reflecting the edges of the graph. The decision variable x_ij is 1 if the edge (i, j) is selected in the tour, and 0 otherwise.

    model.setObjective(quicksum(distance_matrix[i][j] * x_ij[i, j] for i in range(n) for j in range(n) if i != j), GRB.MINIMIZE) ##Objective function is defined as the sum of the distances of the used edges (edges included in HC), which is to be minimized.

    model.addConstrs(quicksum(x_ij[i, j] for j in range(n) if i != j) == 1 for i in range(n))   #Every node must be left exactly once in the tour constraints.
    model.addConstrs(quicksum(x_ij[i, j] for i in range(n) if i != j) == 1 for j in range(n))   #Every node must be visited exactly once.

    model._x_ij = x_ij                  #parameters of the model are defined to be used in the callback function
    model._n = n

    model.optimize(Subtour_Elimination)     #Optimize the model with the callback function for subtour elimination
    
    if model.status in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):              #The results of the optimization model are reported by the return of Gurobi.
        vals = model.getAttr(GRB.Attr.X, x_ij)
        selected = tuplelist((i, j) for i, j in x_ij.keys() if vals[i, j] > 1e-5)     #Tour is obtained by checking which x_ij variables are assigned to ve 1.
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

def Max_N_in_Time(algo, time_limit=600, initial_n=300, step=2, trials=3):   #To find maximum n that can be solved within the time limit, the function takes the algorithm name, time limit,
    n = initial_n           # initial n, step size and number of trials as input. The function returns the maximum n that can be solved within the time limit, but does not necessarily
    while True:             #have to stop when first time limit is reached. This is because omputational times vary experiment to experiment and we tried to forbid outlier cases where the algorithm takes much time.
        exceeded_count = 0  #n is increased by step size when the algorithm is able to solve the problem in defined time limit
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
number_of_vertices = [20, 50, 100, 200, 300]
algorithms = ['NN', 'Christofides', 'TSP IP']

results = []

for n in number_of_vertices:                            #Loop through the number of vertices and algorithms to conduct experiments. For each experimental setting, number_of_experiments amount of experiments are conducted.
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

results_df = pd.DataFrame(results)                #Convert the results list to a DataFrame for easier analysis and export.
summary_df = results_df.groupby(['Number_of_Vertices', 'Algorithm']).agg(  # Group by number of vertices and algorithm to calculate summary statistics. Grouped by algorithm and number of vertices.
    Average_Total_Distance=('Total_Distance', 'mean'),
    Worst_Total_Distance=('Total_Distance', 'max'),
    Average_Runtime=('Algorithm_Time', 'mean'),
    Max_Runtime=('Algorithm_Time', 'max')
).reset_index()

largest_solvable = {algo: Max_N_in_Time(algo, time_limit=600) for algo in algorithms}
largest_solvable_df = pd.DataFrame([{'Algorithm': algo, 'Largest_Solvable_n_within_10min': n} for algo, n in largest_solvable.items()])

with pd.ExcelWriter('tsp_algorithms_results.xlsx') as writer:           #Write results to an xlsx file
    results_df.to_excel(writer, sheet_name='All_Experiments', index=False)
    summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
    largest_solvable_df.to_excel(writer, sheet_name='Largest_Solvable_10min', index=False)


#While conducting experiments, we were unable to run Max_N_in_Time function with small step size since it was taking too long. Therefore, we used a larger step size,
#To find upper and lower bounds on n. Then we iteratively runned the Max_N_in_Time function with smaller step size to find the maximum n that can be solved within the time limit, starting from earlier Max_N_in_Time results.