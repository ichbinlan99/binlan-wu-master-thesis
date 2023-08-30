import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import networkx as nx
import networkx as nx
import numpy as np
import random
import warnings
from causalnex.structure import StructureModel
from causalnex.structure.notears import from_pandas
from causalnex.plots import plot_structure, NODE_STYLE
from causalnex.structure.dynotears import from_pandas_dynamic 
from IPython.display import Image
from scipy.spatial.distance import hamming

class agg_simulation:
    def __init__(self, num_nodes, num_edges, num_run):
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_run = num_run
                
    def generate_random_dag(self):
        if self.num_edges < self.num_nodes - 1:
            raise ValueError("A DAG must have at least N-1 edges for N nodes.")

        # Generate a random graph with num_nodes nodes
        self.G = nx.gn_graph(self.num_nodes)

        # Add additional edges to reach the desired total number of edges
        while self.G.number_of_edges() < self.num_edges:
            u, v = random.sample(range(self.num_nodes), 2)
            # Check if adding this edge would create a cycle
            if not nx.has_path(self.G, u, v):
                self.G.add_edge(u, v)

        return self.G
    
    def perturb_dag(self, dag):
        dag = dag
        new_dag = dag.copy()  # Create a copy of the input DAG to perturb
        nodes = list(new_dag.nodes())

        # Randomly select whether to add or remove an edge
        action = random.choice(['add', 'remove'])

        if action == 'add':
            # Add a new edge while ensuring the graph remains acyclic
            while True:
                u, v = random.sample(nodes, 2)
                if not nx.has_path(new_dag, u, v) and u != v:
                    new_dag.add_edge(u, v)
                    break
        else:  # action == 'remove'
            # Remove an existing edge while ensuring the graph remains acyclic
            if new_dag.number_of_edges() > 0:
                edge_to_remove = random.choice(list(new_dag.edges()))
                new_dag.remove_edge(*edge_to_remove)

        return new_dag
    
    def run_perturb(self, dag):
        initial_dag = dag
        perturbed_adj_matrix = ['' for _ in range(self.num_run)]        
        # Perturb the initial DAG
        for i in range(self.num_run):
            #number of perturbations approved
            num_p = random.choice([0, 1, 2, 3, 4, 5])
            perturbed_dag = initial_dag
            for j in range(num_p):
                perturbed_dag = self.perturb_dag(perturbed_dag)
            # Convert the DAGs to adjacency matrices (0,1 form)
            initial_matrix = nx.adjacency_matrix(initial_dag).toarray()
            perturbed_matrix = nx.adjacency_matrix(perturbed_dag).toarray()

            # Set non-zero entries to 1
            initial_matrix = np.where(initial_matrix > 0, 1, 0)
            perturbed_matrix = np.where(perturbed_matrix > 0, 1, 0)
            perturbed_adj_matrix[i] = perturbed_matrix
        
        return perturbed_adj_matrix, initial_matrix
    
    
    def aggregation(self, perturbed_adj_matrix, initial_matrix, plott=False):
        all_vote = np.array(perturbed_adj_matrix)
        majority_vote_matrix = (np.mean(all_vote, axis=0) > 0.4).astype(int)
        count = np.sum(all_vote, axis = 0)
        cmap = sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True, reverse = True) 
        cmap_1 = sns.cubehelix_palette(start=0, rot=0, as_cmap=True, reverse = True)
        sns.heatmap(count, cmap=cmap, annot=False, fmt="d")
        diff = np.sum(np.abs(majority_vote_matrix - initial_matrix))
        
        if plott:
            plt.figure(figsize=(14, 4))  
            plt.subplot(1, 3, 1)
            sns.heatmap(count, cmap=cmap, annot=False, fmt="d")
            plt.title("Adjacency Matrix by count")

            plt.subplot(1, 3, 2)
            sns.heatmap(majority_vote_matrix, cmap=cmap, annot=False, fmt="d")
            plt.title("Adjacency Matrix after Aggregation")

            plt.subplot(1, 3, 3)
            sns.heatmap(initial_matrix, cmap=cmap_1, annot=False, fmt="d")
            plt.title("Adjacency Matrix of the Global DAG")

            plt.savefig('out/'+'count_5_perturbation.pdf')

            plt.tight_layout()
            plt.show()
            
        return majority_vote_matrix

    def map_to_dag(self, adjacency_matrix):
        # Create a directed graph
        G = nx.DiGraph()
        num_nodes = adjacency_matrix.shape[0]
        G.add_nodes_from(range(num_nodes))

        # Add edges based on the adjacency matrix
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adjacency_matrix[i, j] == 1:
                    G.add_edge(i, j)
                    
        if not nx.has_path(G):
            print("DAG")
            return adjacency_matrix
        else:
            # Convert to a directed acyclic graph
            try:
                dag = nx.dag_longest_path(G)
            except nx.NetworkXNoPath:
                print("No DAG found.")
            else:
                # Create the adjacency matrix of the resulting DAG
                dag_adjacency_matrix = np.zeros((num_nodes, num_nodes))
                for i in range(1, len(dag)):
                    dag_adjacency_matrix[dag[i-1]][dag[i]] = 1
            return dag_adjacency_matrix

class sim_eval:
    
    def __init__(self, matrix):
        self.matrix=matrix 
        
    def majority_voting(self, remaining_matrices):
        consensus_matrix = sum(remaining_matrices) > len(remaining_matrices) / 2
        return consensus_matrix

    def calculate_distance(self, matrix1, matrix2):
        return hamming(matrix1.flatten(), matrix2.flatten())

    def leave_one_out_cross_validation(self, adjacency_matrices):
        distances = []
        for i, validation_matrix in enumerate(adjacency_matrices):
            training_matrices = adjacency_matrices[:i] + adjacency_matrices[i+1:]
            consensus_matrix = self.majority_voting(training_matrices)

            distance = self.calculate_distance(validation_matrix, consensus_matrix)
            distances.append(distance)

        return distances
      
        
def run_sim(num_nodes, num_edges, num_run, plott=False):
    p_dag = agg_simulation(num_nodes, num_edges, num_run) 
    init_dag = p_dag.generate_random_dag()
    perturbed_adj_matrix, init_matrix = p_dag.run_perturb(init_dag)
    agg_matrix = p_dag.aggregation(perturbed_adj_matrix,init_matrix, plott)
    
    return agg_matrix, perturbed_adj_matrix, init_matrix

def run_eval(perturbed_adj_matrix):
    eval_agg = sim_eval(perturbed_adj_matrix)
    # Call the function with a list of adjacency matrices
    adjacency_matrices = perturbed_adj_matrix[1:]  # list of binary adjacency matrices
    distance_results = eval_agg.leave_one_out_cross_validation(adjacency_matrices)

    return distance_results 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    