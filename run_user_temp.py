import pandas as pd
import numpy as np
import tigramite
from tigramite import data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys
from tigramite import plotting as tp
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.lpcmci import LPCMCI
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.regressionCI import RegressionCI
from tigramite.independence_tests.cmiknn import CMIknn
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def get_symbol_with_indices(arr):
    non_empty_indices = np.where(arr != '')  # Get indices where the element is non-empty
    non_empty_arr = arr[non_empty_indices]
    if np.any(arr != ''):
        unique_values, value_counts = np.unique(non_empty_arr, return_counts=True)
        symbol = unique_values[np.argmax(value_counts)]
        indices = np.where(arr == symbol)
        avg_indices = np.floor(np.mean(indices))
#         print(f"'{symbol}' found at average indices: {indices}")
        return symbol, np.floor(avg_indices)
    else:
        return ' ', -0.1
    
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
        pend_graph.append(majority_vote)
        pend_indices.append(edge_ind)
    pend_graph = np.stack(pend_graph)
    pend_indices = np.stack(pend_indices)
    agg_graph = stats.mode(pend_graph, axis=0).mode.squeeze()
    agg_index = np.floor(pend_indices.mean(axis =0))
    
    return agg_graph, agg_index

def stat_to_dy_graph(static_array, lag_array, max_time_lag):
    result_array = np.full(shape = (lag_array.shape[0],lag_array.shape[0],max_time_lag+1), fill_value = '', dtype='<U3')
#     print(result_array)
    for i in range(static_array.shape[0]):
        for j in range(static_array.shape[1]):
            if static_array[i, j] != ' ':
                time = lag_array[i,j]
                if time == 0:
                    pass
                else: 
                    result_array[i, j, int(time)] = static_array[i, j]
#             else:
#                 result_array[i, j, :] = ''
    return result_array


def shuffle_columns(df, n):
    shuffled_dfs = []
    for _ in range(n):
        shuffled_df = df.copy()
        columns = df.columns.tolist()
        np.random.shuffle(columns)
        shuffled_df = shuffled_df[columns]
        shuffled_dfs.append(shuffled_df)

    return shuffled_dfs
def run_user(data, var_names, user_id, cit_method, method_arg, plotting_arg):
    #initialize cit method
    var_names = var_names
    dataframe = pp.DataFrame(data.values, var_names=var_names)
    run_corrlations = False
    plotting_cor = plotting_arg['plotting_cor']
    plotting_ind_graph = plotting_arg['plotting_ind_graph']
    plotting_graph = plotting_arg['plotting_graph']
    method = plotting_arg['method']
#     if not sliding_window:
    if cit_method == "gpdc":
        cit = GPDC(significance='analytic', gp_params=None)
    elif cit_method == "cmi":
        cit = CMIknn(significance='shuffle_test', knn=0.1, shuffle_neighbors=5, transform='ranks', sig_samples=200)
    elif cit_method == "parcorr":
        cit=ParCorr()
    else:
        raise ValueError('cit method not supported.')
    if method == 'pcmci':
        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=cit,
            verbosity=0)
    else:
        pcmci =  LPCMCI(
            dataframe=dataframe, 
            cond_ind_test=cit,
            verbosity=0)

    if run_corrlations:
        correlations = pcmci.run_bivci(tau_max=method_arg['tau_max'], val_only=True)['val_matrix']
        if plotting_cor == True:
            lag_func_matrix = tp.plot_lagfuncs(val_matrix=correlations, save_name = 'out/corr.pdf',
                                   setup_args={'var_names':var_names, 'figsize':(15, 10),
                                    'x_base':5, 'y_base':.5})
            matrix_lags = np.argmax(np.abs(correlations), axis=2)
            tp.plot_densityplots(dataframe=dataframe, setup_args={'figsize':(15, 10)}, save_name = 'out/density.pdf', add_densityplot_args={'matrix_lags':matrix_lags}) 
            plt.show()
        
    tau_max = method_arg['tau_max']
    sig_level = method_arg['sig_level']#[0.2]#[None, 0.05, 0.1, 0.2, 0.3]
    n = method_arg['shuffle_times']
    if n > 1:
        results_graph = []
        results_val = []
        for pc_alpha in sig_level:
            for i in range(n):
                link_assumptions = method_arg['link_assumptions']
                shuffled_dataframes = shuffle_columns(data, n)
                shuffled = pp.DataFrame(shuffled_dataframes[i].values, var_names = shuffled_dataframes[i].columns)
                ind = shuffled_dataframes[i].columns.tolist().index('bld_vol')
                link_assumptions[ind] = {} 
                pcmci_shuffle = PCMCI(
                    dataframe=shuffled,
                    cond_ind_test=cit,
                    verbosity=0)
                res = pcmci_shuffle.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=pc_alpha, link_assumptions=link_assumptions)
                reverted_indices = [list(shuffled_dataframes[i].columns.tolist()).index(col) for col in data.columns]
            # Revert the order of both axes of the array using the column indices
                res['graph'] =  res['graph'][reverted_indices][:,reverted_indices]
                res['val_matrix'] = res['val_matrix'][reverted_indices][:,reverted_indices]
                results_graph.append(res['graph'])
                if plotting_ind_graph:
                    tp.plot_graph(res['graph'], val_matrix=res['val_matrix'], figsize = (20, 10),
                      save_name = 'out/'+str(user_id)+ '_'+ str(cit_method)+ '_shuffle' + str(i)+'th_'+str(pc_alpha)+ '_'+ str(tau_max)+ '.svg', show_colorbar=True, 
                      node_size = 0.3, label_fontsize=20, node_label_size=18, link_label_fontsize=18, 
                      curved_radius=0.3, var_names=var_names)    
#                 results_val.append(res['val_matrix'])
            graph, index = static_graph_with_lag_info(results_graph, data.shape[1])
            aggregated_array = stat_to_dy_graph(graph, index, tau_max+1)
            tp.plot_graph(aggregated_array, figsize = (20, 10), save_name = 'out/'+'shuffled_agg_'+str(user_id)+ '_' +str(n)+'_'+str(pc_alpha)+'_'+str(tau_max)+'.svg', 
                      show_colorbar=True, node_size = 0.3, label_fontsize=20, node_label_size=18, 
                      link_label_fontsize=18, curved_radius=0.3, var_names=var_names)
        return results_graph
    else:
        all_res = []
        for pc_alpha in sig_level:
            link_assumptions = method_arg['link_assumptions']
            ind = data.columns.tolist().index('bld_vol')
            results = pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=pc_alpha, link_assumptions=link_assumptions)
            all_res.append(results)
            if plotting_graph:
                tp.plot_graph(results['graph'], val_matrix=results['val_matrix'], figsize = (20, 10),
                          save_name = 'out/'+str(user_id) + '_'+ str(cit_method)+'_'+str(pc_alpha)+ str(tau_max)+ '.pdf', show_colorbar=True, 
                          node_size = 0.3, label_fontsize=20, node_label_size=18, link_label_fontsize=18, 
                          curved_radius=0.3, var_names=var_names)
        return all_res
