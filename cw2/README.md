# DGL2024 Brain Graph Super-Resolution Challenge

## Contributors

Team name: mine bitcoins dont train models
Team members:
* Mihnea Ghitu
* Tao Wang 
* Stefan Savulescu 
* Andrei Tirziu

## Problem Description

- A short description of the problem you are solving. Why is the brain graph super-resolution an ineresting problem to solve?

## MASTGNN - Methodology

<!-- - Summarize in a few sentences the building blocks of your generative GNN model. -->

- Model Architecture
![Model Architecture](/GSR-Net/images/arch.png)

Our model consists of:
1) Graph U-Net Block: This component is used to under-
stand the key topological features of our low-resolution
graphs in order to generate high-resolution counterparts
that are similar in structure. It contains 5 down-
sampling and 5 up-sampling levels, each of them
with their own pool / unpool layer and a convolution
layer, as well as 3 more convolution layers added at
the start, end and in the middle of the latent (bottom)
component.
2) GAT Layers: As an improvement, we replaced all
normal Graph Convolution (GCN) in the original GSR architecture layers with Graph
Attention layers (as introduced by ). These use the
self-attention mechanism, which computes the impor-
tance of a certain node within a graph. Therefore, using
GAT layers within U-Net makes them more efficient by
differentiating between the most important topological
properties and thus making it easier for the pooling
layers to extract them.
3) Residual Connections: We recognised
that, were the super-resolution objective higher (i.e. if
we needed to predict graphs with even more nodes), our
architecture needed to accommodate for a higher number
of convolution layers. Since it is well known that gra-
dient magnitude is inversely proportional to the number
of layers used (i.e. the complexity of the architectures),
each of our newly enhanced layers now contains 2
convolution layers with one residual connection that
consists of the input initially given to the new type of
layer. In this way, we allow for the gradient to flow
through the whole network, without reaching values of
0

## Results

- 3 Fold Cross-Validation metrics
![3 Fold Cross-Valitation](/GSR-Net/images/Metrics.png)

# Reproducibility

To run our proposed model, one can go to GSR-Net folder where all our source code is saved and run the file **pipeline.py**.

This will firstly define the appropriate device (CPU or cuda) to run the model and anchor the random seed for reproducibility of the results (from ./GSR-Net/globals.py)

It also contains a flag for producing the plots (which currently is set to True, so the evaluation metrics for the 3 folds are displayed at the end of the training). This was done because of the long time needed to compute the centrality measurements (around 15 minutes per fold).

The current implementation runs the 3 Fold CV with the provided LR and HR datasets. It also provides the ability to run the fine tuned model on the entire dataset - to do this one should uncomment the line with **train_all_dataset()** and comment the **run_experiment()** line (both at the end of pipeline.py).
