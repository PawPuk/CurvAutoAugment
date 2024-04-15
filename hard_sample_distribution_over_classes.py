import argparse
import pickle
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def load_pickle_file(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def calculate_statistics(counts_per_class):
    means = [np.mean(counts) if counts else np.nan for counts in counts_per_class]
    stds = [np.std(counts) if counts else np.nan for counts in counts_per_class]
    return means, stds


def print_statistics(means, stds, description):
    print(description)
    for i, (mean, std) in enumerate(zip(means, stds)):
        print(f"Class {i}: Mean = {mean:.2f}, Std = {std:.2f}")


def plot_mean_std_bar_chart(means, stds, dataset, strategy, x_label='Class', y_label='Number of Hard Samples'):
    fig, ax = plt.subplots(figsize=(10, 6))
    indices = list(range(len(means)))
    ax.bar(indices, means, yerr=stds, capsize=5, color='skyblue', error_kw={'elinewidth': 2, 'ecolor': 'black'})
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_xticks(indices)
    ax.set_xticklabels([f'Class {i}' for i in indices], rotation=45)
    ax.grid(True, linestyle='--', which='both', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'Figures/Figure 8/hard_sample_distribution_{strategy}_{dataset}.pdf')
    plt.savefig(f'Figures/Figure 8/hard_sample_distribution_{strategy}_{dataset}.png')


def main(dataset_name: str, strategy: str):
    filepaths = [
        f'Results/F1_results/{dataset_name}_{strategy}_False_20000_indices.pkl',
        f'Results/F1_results/{dataset_name}_{strategy}_True_20000_indices.pkl'
    ]
    aggregated_counts = [[] for _ in range(10)]  # Assuming 10 classes

    for filepath in filepaths:
        data = load_pickle_file(filepath)
        # We will collect counts as list of integers for all experiments per class
        counts_per_class = [[len(tensor) for tensor in experiment] for experiment in zip(*data)]
        means, stds = calculate_statistics(counts_per_class)
        print_statistics(means, stds, f"Statistics for {filepath}:")

        # Extend aggregated counts with the new counts
        for i, class_counts in enumerate(counts_per_class):
            aggregated_counts[i].extend(class_counts)

    aggregated_means, aggregated_stds = calculate_statistics(aggregated_counts)
    print_statistics(aggregated_means, aggregated_stds, "Aggregated statistics:")
    plot_mean_std_bar_chart(aggregated_means, aggregated_stds, dataset_name, strategy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script to investigate the impact of reducing hard/easy samples on generalisation.')
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        help='Dataset name. The code was tested on MNIST, FashionMNIST and KMNIST.')
    parser.add_argument('--strategy', type=str, choices=['stragglers', 'confidence', 'energy'],
                        default='stragglers', help='Strategy (method) to use for identifying hard samples.')
    args = parser.parse_args()
    main(**vars(args))
