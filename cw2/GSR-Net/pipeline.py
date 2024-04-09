import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from globals import TORCH_DEVICE, COMPUTE_METRICS_FLAG, FULL_DATA, FULL_TARGETS
from hyperparams import Hyperparams
from model import GSRNet
from preprocessing import get_unseen_test_dset, RANDOM_SEED
from sklearn.model_selection import KFold
from utils import pad_HR_adj, unpad
from metrics import compute_metrics,  plot_metrics_per_type
from MatrixVectorizer import MatrixVectorizer

# Vectorize a list of HR graphs (either predictions or ground truth)
# and concatenates the values from all graphs
def vectorize_adj_list(adj_list):
    vect_adj = []
    vectorizer = MatrixVectorizer()

    for adj in adj_list:
        vect_adj.append(vectorizer.vectorize(adj))
    vect_adj = np.concatenate(vect_adj)
    
    return vect_adj

# Apply post-processing on the model outputs
# We want our predictions to all be in [0, 1] and our model returns only
# positive values, so we only upper bound our predictions to 1
def postprocessing(preds):
    final_preds = torch.where(preds > 1, 1, preds)
    return final_preds

def train(model, optimizer, subjects_adj, subjects_labels, hps):
    model.train()
    all_epochs_loss = []
    no_epochs = hps.epochs

    epoch_loss = []
    epoch_error = []

    min_loss, min_loss_epoch = 1e20, -1
    for epoch in range(no_epochs):
        epoch_loss = []
        epoch_error = []

        for group in optimizer.param_groups:
            group['lr'] *= hps.lr_schedule

        # Generate the permutation to apply to the train dataset
        permutation = torch.randperm(hps.lr_dim)
        perm = torch.zeros((hps.lr_dim, hps.lr_dim))

        for i in range(hps.lr_dim):
            perm[i][permutation[i]] = 1

        perm = perm.to(torch.double)
        perm_t = perm.transpose(0, 1)

        # Get the permuted adjacency matrices
        new_adj = torch.matmul(perm_t, torch.matmul(torch.from_numpy(subjects_adj),
                                                    perm)).numpy()

        for lr, hr in zip(new_adj, subjects_labels):
            model.train()
            optimizer.zero_grad()

            lr = torch.from_numpy(lr).type(torch.FloatTensor).to(TORCH_DEVICE)
            hr = torch.from_numpy(hr).type(torch.FloatTensor).to(TORCH_DEVICE)

            model_outputs, net_outs, start_gcn_outs, _ = model(lr)
            model_outputs = postprocessing(model_outputs)
            # Unpad the 320x320 graph to 268x268
            model_outputs  = unpad(model_outputs, hps.padding)

            # Get the eigevectors of the padded 320x320 graph for the GT HR
            padded_hr = pad_HR_adj(hr,hps.padding)
            _, upper_hr = torch.linalg.eigh(padded_hr, UPLO='U')

            # Compute the final loss which represents the sum of:
            # * The reconstruction loss of the decoded graph through the UNet
            # * The eigen loss between our modelled weight matrix 
            # and the real eigenvectors of the HR
            # * And the final super resolution loss between our modelled output
            # and the ground truth HR graph
            loss = hps.lmbda * hps.train_criterion(net_outs, start_gcn_outs) + \
                   hps.train_criterion(model.layer.weights,upper_hr) + \
                   hps.train_criterion(model_outputs, hr)

            # Record the super resolution loss for logging purposes
            error = hps.train_criterion(model_outputs, hr)
            loss_val = loss.item()

            # Apply the backpropagation with the previously computed loss
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss_val)
            if loss_val < min_loss:
                min_loss = loss_val
                min_loss_epoch = epoch
                # torch.save(model.state_dict(), 'models/GSR_model.pth')
            epoch_error.append(error.item())

        # Logs remained from saving the errors on wandb
        # wandb.log({"loss": np.mean(epoch_loss), "error": np.mean(epoch_error)*100})
        print("Epoch: ", epoch, "Loss: ", np.mean(epoch_loss), "Error: ", np.mean(epoch_error)*100,"%")
        all_epochs_loss.append(np.mean(epoch_loss))
    print("Min loss: ", min_loss, "Min loss epoch: ", min_loss_epoch)
    # wandb.log({"min_loss": min_loss, "min_loss_epoch": min_loss_epoch})

def get_mae_from_vect_preds(preds_list, test_labels, fold_num):
    # vectorize both the predictions and the ground truth graphs
    vect_hr = vectorize_adj_list(test_labels)
    vect_preds = vectorize_adj_list(preds_list)

    df = pd.DataFrame({
        'ID': np.arange(1, len(vect_preds) + 1),
        'Predicted': vect_preds
    })
    # save the predictions
    df.to_csv('predictions_fold_{}.csv'.format(fold_num), index=False)
    # compute the MAE as done for the submissions
    mean_mae = np.mean(np.absolute(vect_preds - vect_hr))

    return mean_mae

def test(model, test_adj, test_labels, hps, num_fold):
    model.eval()
    test_error = []
    preds_list=[]
    g_t = []

    for lr, hr in zip(test_adj,test_labels):
        all_zeros_lr = np.any(lr)
        all_zeros_hr = np.any(hr)
        if all_zeros_lr and all_zeros_hr:
            lr = torch.from_numpy(lr).type(torch.FloatTensor).to(TORCH_DEVICE)
            np.fill_diagonal(hr,1)
            hr = torch.from_numpy(hr).type(torch.FloatTensor).to(TORCH_DEVICE)
            preds, _, _, _ = model(lr)
            preds = postprocessing(preds)
            preds = unpad(preds, hps.padding)

            preds_list.append(preds.detach().clone().cpu().numpy())

            error = F.l1_loss(preds, hr)
            g_t.append(hr.flatten())
            print(error.item())
            # wandb.log({"test_error_item": error.item()})
            test_error.append(error.item())

    mean_mae = get_mae_from_vect_preds(preds_list, test_labels, num_fold)

    print ("Test error MAE: ", mean_mae)
    # wandb.log({"mean_mae": mean_mae})

    if COMPUTE_METRICS_FLAG:
        return mean_mae, compute_metrics(preds_list, test_labels, hps)
    else:
        return mean_mae, {}

# Function that runs the 3F CV
def run_experiment(X, Y):

    cv = KFold(n_splits=3, random_state=RANDOM_SEED, shuffle=True)
    hps = Hyperparams()
    eval_metrics_list = []

    best_model, best_loss = None, 100
    for num_fold, (train_index, test_index) in enumerate(cv.split(X)):
        model = GSRNet(hps)
        model.to(TORCH_DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=hps.lr)
        subjects_adj, test_adj, subjects_ground_truth, test_ground_truth = X[
            train_index], X[test_index], Y[train_index], Y[test_index]

        train(model, optimizer, subjects_adj, subjects_ground_truth, hps)
        mae_loss, eval_metrics = test(model, test_adj, test_ground_truth,
                                      hps, num_fold + 1)
        eval_metrics_list.append(eval_metrics)

        if COMPUTE_METRICS_FLAG:
            print("Metrics for this fold:")
            for key, val in eval_metrics.items():
                print(key, val)

        if mae_loss < best_loss:
            best_model = model

    torch.save(best_model.state_dict(), 'models/GSR_best.pth')

    if COMPUTE_METRICS_FLAG:
        plot_metrics_per_type(eval_metrics_list)

def get_unseen_predictions(model, hps):
    model.eval()
    unseen_test = get_unseen_test_dset()
    preds_list = []

    for lr_adj in unseen_test:
        all_zeros_lr = not np.any(lr_adj)

        if all_zeros_lr == False: #choose representative subject
            lr_adj = torch.from_numpy(lr_adj).type(torch.FloatTensor)
            preds = model(lr_adj)[0]
            preds = postprocessing(preds)
            preds = unpad(preds, hps.padding)

        preds_list.append(preds.detach().cpu())

    res = vectorize_adj_list(preds_list)
    df = pd.DataFrame({
        'ID': np.arange(1, len(res) + 1),
        'Predicted': res
    })

    df.to_csv('submission.csv', index=False)

# Function to run the best model on entire dataset
def train_on_entire_dataset(X, Y):
    hps = Hyperparams()

    model = GSRNet(hps)
    model.to(TORCH_DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=hps.lr)
    subjects_adj, subjects_ground_truth = X, Y
    # train model on entire dataset
    train(model, optimizer, subjects_adj, subjects_ground_truth, hps)

    # save the trained model
    torch.save(model.state_dict(), 'models/GSR_best.pth')

    # get the predictions on the test dataset
    get_unseen_predictions(model, hps)

run_experiment(FULL_DATA, FULL_TARGETS)
# train_on_entire_dataset(FULL_DATA, FULL_TARGETS)