import numpy as np
from scipy import stats
import math

def get_symbol_with_indices(arr):
    non_empty_indices = np.where(arr != '')  # Get indices where the element is non-empty
#     print(non_empty_indices)
    non_empty_arr = arr[non_empty_indices]
    if np.any(arr != ''):
        unique_values, value_counts = np.unique(non_empty_arr, return_counts=True)
        symbol = unique_values[np.argmax(value_counts)]
        indices = np.where(arr == symbol)
        avg_indices = np.floor(np.mean(indices))
#         print(f"'{symbol}' found at average indices: {indices}")
        return symbol, np.floor(avg_indices)
    else:
        return ' ', None
    
def static_graph_with_lag_info(list_graph: list, num_variables):
#     list_graph = [results_497['graph'], results_372['graph']]
    ind = np.empty((num_variables,num_variables))
    pend_graph = []
    pend_indices = []
    for graph in list_graph:      
        majority_vote = np.empty((num_variables,num_variables), dtype='<U3')
        edge_ind = np.empty((num_variables,num_variables))
        for i in range(num_variables):
            for j in range(num_variables):
                majority_vote[i,j], edge_ind[i,j] = get_symbol_with_indices(graph[i,j])
#                 print(i,j, majority_vote[0,2], edge_ind[0,2])
        pend_graph.append(majority_vote)
        pend_indices.append(edge_ind)
    pend_graph = np.stack(pend_graph)
    pend_indices = np.stack(pend_indices)
    agg_graph = stats.mode(pend_graph, axis=0).mode.squeeze()
    new_ind = np.full((num_variables,num_variables), np.nan)
    for i in range(num_variables):
        for j in range(num_variables):
            ind = []
            for k in range(len(list_graph)):
                if pend_graph[k][i,j] != '':
                    ind.append(pend_indices[k][i,j])
            not_nan = [item for item in ind if not math.isnan(item)]
#             print(not_nan, i,j)
            new_ind[i,j] = np.mean(not_nan)
    agg_index = np.floor(new_ind)
    
    return agg_graph, agg_index

def stat_to_dy_graph(static_array, lag_array, max_time_lag):
    result_array = np.full(shape = (lag_array.shape[0],lag_array.shape[0],max_time_lag+1), fill_value = '', dtype='<U3')
#     print(result_array)
    for i in range(static_array.shape[0]):
        for j in range(static_array.shape[1]):
            if static_array[i, j] != ' ':
                time = lag_array[i,j]
#                 if time == 0:
#                     pass
#                 else: 
                result_array[i, j, int(time)] = static_array[i, j]
#             else:
#                 result_array[i, j, :] = ''
    for i in range(result_array.shape[0]):
        for j in range(i, result_array.shape[0]):
            link = result_array[:,:,0][i,j]
    #         print(link)
            if link == '-->':
                reverse_link = '<--'
            elif link == '<--':
                reverse_link = '-->'
            elif link == 'o-o':
                reverse_link = 'o-o'
            elif link == 'x-x':
                reverse_link = 'x-x'
            else:
                reverse_link = ' '.join(reversed(link))
            result_array[:,:,0][j,i] = reverse_link
    return result_array


def get_adjacency_matrix(graph):
    # Initialize the adjacency matrix with zeros
    adjacency_matrix = np.zeros((graph.shape[0], graph.shape[0]), dtype=int)

    # Convert symbols to binary values in the adjacency matrix
    for i in range(graph.shape[0]):
        for j in range(graph.shape[0]):
            for t in range(graph.shape[2]):
                if graph[i][j][t] == '-->':
                    adjacency_matrix[i, j] = 1
                elif graph[i][j][t] == 'o-o':
                    adjacency_matrix[i, j] = 0.5
                elif graph[i][j][t] == 'x-x':
                    adjacency_matrix[i, j] = 0.5
    return adjacency_matrix
