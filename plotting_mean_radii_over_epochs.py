import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch


def load_radii_data(filepath='Results/Radii_over_epoch/MNIST_all_epoch_radii.pkl'):
    with open(filepath, 'rb') as f:
        all_epoch_radii = pickle.load(f)
    return all_epoch_radii


def compute_statistics(all_epoch_radii):
    # Extract number of experiments and epochs
    num_experiments = len(all_epoch_radii)
    num_epochs = len(all_epoch_radii[0])  # Assuming all experiments have the same number of epochs
    num_classes = 10  # Number of classes

    # Initialize containers for radii values
    radii_values = np.zeros((num_classes, num_epochs, num_experiments))

    for experiment_idx, experiment_data in enumerate(all_epoch_radii):
        for epoch_idx, (epoch, class_radii_dict) in enumerate(experiment_data):
            for class_id, radius in class_radii_dict.items():
                radii_values[class_id, epoch_idx, experiment_idx] = radius.item()

    # Compute mean and standard deviation across experiments for each class and epoch
    mean_radii = np.mean(radii_values, axis=2)
    std_radii = np.std(radii_values, axis=2)

    return mean_radii, std_radii


def plot_mean_std_radii(mean_radii, std_radii, dataset_name):
    epochs = np.arange(21, 21 + mean_radii.shape[1])  # Starting from epoch 21
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    for class_id in range(mean_radii.shape[0]):
        ax = axes[class_id // 5, class_id % 5]
        ax.plot(epochs, mean_radii[class_id], label='Mean Radius')
        ax.fill_between(epochs, mean_radii[class_id] - std_radii[class_id], mean_radii[class_id] + std_radii[class_id], alpha=0.5, label='Std. Dev.')
        ax.set_title(f'Class {class_id}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Radius')

    plt.suptitle(f'Radii Development Over Epochs for {dataset_name}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'Figures/Figure 2/mean_std_radii_on_{dataset_name}.png')
    plt.savefig(f'Figures/Figure 2/mean_std_radii_on_{dataset_name}.pdf')
    plt.show()


if __name__ == "__main__":
    dataset_name = 'MNIST'  # Adjust as needed
    all_epoch_radii = load_radii_data()
    mean_radii, std_radii = compute_statistics(all_epoch_radii)
    plot_mean_std_radii(mean_radii, std_radii, dataset_name)
