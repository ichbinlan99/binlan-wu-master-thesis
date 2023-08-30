from tigramite.toymodels import structural_causal_processes as toys
from tigramite import data_processing as pp
from tigramite import plotting as tp
import numpy as np
import matplotlib.pyplot as plt

seed = 7
# auto_coeff = 0.95
# coeff = 0.4
T = 180
def lin(x): return x

links ={0: [((0, -1), 0.02, lin),
            ((0, -2), 0.02, lin),
            ((0, -3), 0.02, lin),
            ((0, -4), 0.02, lin),
            ((0, -5), 0.02, lin),
            ((0, -6), 0.02, lin),
            ((6, -4), 0.3, lin)
            ],
        1: [((1, -1), 0.05, lin),
            ((1, -2), 0.05, lin),
            ((1, -3), 0.05, lin),
            ((1, -4), 0.05, lin),
            ((1, -5), 0.05, lin),
            ((1, -6), 0.05, lin),
            ((6, -4), 0.3, lin)
            ],
        2: [((2, -1), 0.025, lin), 
            ((2, -2), 0.025, lin), 
            ((2, -3), 0.025, lin), 
            ((2, -4), 0.025, lin), 
            ((2, -5), 0.025, lin), 
            ((2, -6), 0.025, lin), 
            ((0, 0), 0.75, lin), 
            ],
        3: [((3, -1), 0.3, lin), 
            ((3, -2), 0.3, lin),
            ((3, -3), 0.3, lin),
            ((3, -4), 0.3, lin),
            ((2, -1), 0.225, lin), 
            ],
        4: [((4, -1), 0.27, lin), 
            ((4, -2), 0.27, lin), 
            ((4, -3), 0.27, lin), 
            ((4, -4), 0.27, lin), 
            ((4, -5), 0.27, lin),
            ((4, -6), 0.27, lin), 
            ((7, -3), 0.01, lin), 
            ],   
        5: [((5, -1), 0.3, lin), 
            ((5, -2), 0.3, lin),
            ((5, -3), 0.3, lin),
            ((5, -4), 0.3, lin),
            ((5, -1), 0.225, lin), 
            ((3, -2), 0.2, lin), 
            ],  
        6: [((6, -1), 0.5, lin), 
            ((6, -2), 0.5, lin),],  
        7: [((7, -1), 2.0, lin), 
            ((7, -2), 2.0, lin),
            ((7, -3), 2.0, lin), 
            ((7, -4), 2.0, lin), 
            ((4, 0), 0.1, lin), 
            ((0, -1), 0.0005, lin)
            ]
       }
# Specify dynamical noise term distributions, here unit variance Gaussians
random_state = np.random.RandomState(seed)
noises = [random_state.randn for j in links.keys()]
    
data, nonstationarity_indicator = toys.structural_causal_process(
    links=links, T=T, noises=noises, seed=seed)
T, N = data.shape

# Initialize dataframe object, specify variable names
var_names = ["sbp",
            "dbp",
            "heart_rate",
            "resp_rate",
            "temp_skin",
            "spo2",
            "bld_vol",
            "eda"]

def synthetic_data():
    sy_data = data
    return sy_data

true_graph = toys.links_to_graph(links=links)

def plot_gt(plot_time=True):
    fig = tp.plot_graph(true_graph, show_colorbar=False, save_name='out/gt.pdf', figsize=(10,10),
                    node_size = 0.3, label_fontsize=20, node_label_size=18, link_label_fontsize=25, 
                      curved_radius=0.1,
                     var_names=var_names)
    if plot_time:
        fig = tp.plot_time_series_graph(
        figsize=(10, 10),
        node_size=0.01,
        graph=true_graph,
        var_names=var_names,
        save_name = 'out/gt_time.svg',
        label_fontsize=17.5, 
        )
    plt.show()
    return true_graph