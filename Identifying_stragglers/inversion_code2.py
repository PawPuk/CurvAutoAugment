import matplotlib.pyplot as plt
import numpy as np
import torch

import inversion_code1

#################################
# MODEL AND TRAINING PARAMETERS #
#################################
DEPTH = 2
WIDTH = 20
LATENT = 1  # ordinal number of hidden layer where the observables are computed
N_RUNS = 3  # number of runs from independent initializations
EPOCHS = 150
LEARNING_RATE = 0.1
OPTIMIZER = torch.optim.SGD
#################################

# Load data
data, labels, test_data, test_labels = inversion_code1.load_data(inversion_code1.DATA_BLOCK)
input_size = data.shape[2] * data.shape[3]  # 32*32 for CIFAR, 28*28 for *MNIST

# Setup lists for results
radii_data, distances_data, losses_data = [], [], []
results = []

# Perform N_RUNS independent training runs
for niter in range(N_RUNS):
    model = inversion_code1.NN_KHL(input_size, DEPTH, WIDTH, LATENT).to(inversion_code1.device)
    optimizer = OPTIMIZER(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    # Train
    results_run, _ = inversion_code1.train_and_measure(model, data, labels, test_data, test_labels, optimizer,
                                                       criterion, EPOCHS)
    results.append(results_run)

# Plot results
arr_res = np.array(results)
for kk in range(len(arr_res)):
    plt.plot(arr_res[kk, :, 0], arr_res[kk, :, 3], color="#3a5a4070")  # first radius VS training error
    plt.plot(arr_res[kk, :, 0], arr_res[kk, :, 4], color="#67671570")  # second radius VS training error
    plt.plot(arr_res[kk, :, 0], arr_res[kk, :, 5], color="#06467570")  # distance VS training error
plt.show()
