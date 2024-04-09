from MatrixVectorizer import MatrixVectorizer
import numpy as np
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
import networkx as nx
import matplotlib.pyplot as plt

def compute_metrics(preds_list, target_list, hps):

    # Initialize lists to store MAEs for each centrality measure
    mae_bc = []
    mae_ec = []
    mae_pc = []

    pred_1d_list = []
    gt_1d_list = []

    # Iterate over each test sample
    for i in range(len(target_list)):
        # Convert adjacency matrices to NetworkX graphs
        pred_graph = nx.from_numpy_array(preds_list[i])
        gt_graph = nx.from_numpy_array(target_list[i])

        # Compute centrality measures
        pred_bc = nx.betweenness_centrality(pred_graph, weight="weight")
        pred_ec = nx.eigenvector_centrality(pred_graph, weight="weight")
        pred_pc = nx.pagerank(pred_graph, weight="weight")

        gt_bc = nx.betweenness_centrality(gt_graph, weight="weight")
        gt_ec = nx.eigenvector_centrality(gt_graph, weight="weight")
        gt_pc = nx.pagerank(gt_graph, weight="weight")

        # Convert centrality dictionaries to lists
        pred_bc_values = list(pred_bc.values())
        pred_ec_values = list(pred_ec.values())
        pred_pc_values = list(pred_pc.values())

        gt_bc_values = list(gt_bc.values())
        gt_ec_values = list(gt_ec.values())
        gt_pc_values = list(gt_pc.values())

        # Compute MAEs
        mae_bc.append(mean_absolute_error(pred_bc_values, gt_bc_values))
        mae_ec.append(mean_absolute_error(pred_ec_values, gt_ec_values))
        mae_pc.append(mean_absolute_error(pred_pc_values, gt_pc_values))

        # Vectorize matrices
        pred_1d_list.append(MatrixVectorizer.vectorize(preds_list[i]))
        gt_1d_list.append(MatrixVectorizer.vectorize(target_list[i]))

    # Compute average MAEs
    avg_mae_bc = sum(mae_bc) / len(mae_bc)
    avg_mae_ec = sum(mae_ec) / len(mae_ec)
    avg_mae_pc = sum(mae_pc) / len(mae_pc)

    # Concatenate flattened matrices
    pred_1d = np.concatenate(pred_1d_list)
    gt_1d = np.concatenate(gt_1d_list)

    # Compute metrics
    mae = mean_absolute_error(pred_1d, gt_1d)
    pcc = pearsonr(pred_1d, gt_1d)[0]
    js_dis = jensenshannon(pred_1d, gt_1d)

    return {"MAE" : mae, "PCC" : pcc, "JSD" : js_dis,
            "MAE (PC)" : avg_mae_bc,
            "MAE (EC)" : avg_mae_ec,
            "MAE (BC)" : avg_mae_pc}

def get_average_metrics(metrics_per_fold, labels):
    average_metrics = {}

    for key in labels:
        sum = 0
        for i in range(len(metrics_per_fold)):
            sum += metrics_per_fold[i][key]
        average_metrics[key] = sum / len(metrics_per_fold)

    return average_metrics
    

def plot_metrics_per_fold(metrics_per_fold):
    labels = metrics_per_fold[0].keys()
    average_metrics = get_average_metrics(metrics_per_fold, labels)

    fig, axs = plt.subplots(2, 2, figsize = (10, 5))

    axs[0, 0].bar(labels, [metrics_per_fold[0][label] for label in labels])
    axs[0, 0].set_title('Fold 1')

    axs[0, 1].bar(labels, [metrics_per_fold[1][label] for label in labels])
    axs[0, 1].set_title('Fold 2')

    axs[1, 0].bar(labels, [metrics_per_fold[2][label] for label in labels])
    axs[1, 0].set_title('Fold 3')

    axs[1, 1].bar(labels, [average_metrics[label] for label in labels])
    axs[1, 1].set_title('Avg. Across Folds')

    plt.tight_layout()

    plt.show()

def plot_metrics_per_type(metrics_per_fold):
    labels = metrics_per_fold[0].keys()
    average_metrics = get_average_metrics(metrics_per_fold, labels)

    num_columns = 3
    num_rows = 2

    fig, axs = plt.subplots(num_rows, num_columns, figsize = (10, 5))

    posx = 0
    posy = 0
    
    for label in labels:
        axs[posx, posy].set_title(label)

        for split in range(len(metrics_per_fold)):
            axs[posx, posy].bar('Fold {}'.format(split),
                                metrics_per_fold[split][label])
        
        axs[posx, posy].bar('Avg. Folds',
                            average_metrics[label])
        
        posy += 1
        if posy == num_columns:
            posx += 1
            posy = 0

    plt.tight_layout()

    plt.show()