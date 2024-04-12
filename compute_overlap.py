import pickle
import numpy as np
import torch


def load_and_aggregate_tensors(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    # Aggregate tensors for each experiment into a single tensor
    aggregated_per_experiment = [torch.cat([experiment[i] for i in range(len(experiment))]) for experiment in data]
    return aggregated_per_experiment


"""def compute_overlap_percentages(tensor_list1, tensor_list2):
    # Compute overlaps as percentages of the number of stragglers in tensor_list1
    percentages = [
        len(np.intersect1d(t1.cpu().numpy(), t2.cpu().numpy(), assume_unique=True)) / len(np.unique(t1.cpu().numpy())) * 100
        if len(np.unique(t1.cpu().numpy())) > 0 else np.nan  # Avoid division by zero; mark as NaN if t1 has no elements
        for t1, t2 in zip(tensor_list1, tensor_list2)
    ]
    return percentages


def calculate_and_print_overlap_statistics(aggregated_data_1, aggregated_data_2, strategy_pair):
    overlap_percentages = compute_overlap_percentages(aggregated_data_1, aggregated_data_2)
    mean_percentage = np.nanmean(overlap_percentages)  # Use nanmean to ignore NaN values
    std_percentage = np.nanstd(overlap_percentages)  # Use nanstd to ignore NaN values
    print(f"Overlap percentage between {strategy_pair}: Mean = {mean_percentage:.2f}%, Std = {std_percentage:.2f}%")"""


def compute_overlap_percentages_relative_to_energy(tensor_list1, tensor_list2):
    # Compute overlaps as percentages of the number of indices in tensor_list2 (energy strategy)
    percentages = [
        len(np.intersect1d(t1.cpu().numpy(), t2.cpu().numpy(), assume_unique=True)) / len(np.unique(t2.cpu().numpy())) * 100
        if len(np.unique(t2.cpu().numpy())) > 0 else np.nan  # Avoid division by zero; mark as NaN if t2 has no elements
        for t1, t2 in zip(tensor_list1, tensor_list2)
    ]
    return percentages


def calculate_and_print_overlap_statistics_relative_to_energy(aggregated_data_1, aggregated_data_2, strategy_pair):
    overlap_percentages = compute_overlap_percentages_relative_to_energy(aggregated_data_1, aggregated_data_2)
    mean_percentage = np.nanmean(overlap_percentages)  # Use nanmean to ignore NaN values
    std_percentage = np.nanstd(overlap_percentages)  # Use nanstd to ignore NaN values
    print(f"Overlap percentage relative to 'energy' between {strategy_pair}: Mean = {mean_percentage:.2f}%, Std = {std_percentage:.2f}%")


strategies = ['stragglers', 'energy', 'confidence']
filepaths = [f'Results/F1_results/MNIST_{strategy}_False_20000_indices.pkl' for strategy in strategies]

# Load and aggregate data from each file
aggregated_data_per_strategy = [load_and_aggregate_tensors(filepath) for filepath in filepaths]
print(aggregated_data_per_strategy[0])
print(len(aggregated_data_per_strategy[0]))
print(len(aggregated_data_per_strategy[1]))
print(len(aggregated_data_per_strategy[2]))
print([t.shape for t in aggregated_data_per_strategy[0]])
print([t.shape for t in aggregated_data_per_strategy[1]])
print([t.shape for t in aggregated_data_per_strategy[2]])


# Calculate overlaps and print statistics
strategy_pairs = [('stragglers', 'energy'), ('stragglers', 'confidence'), ('energy', 'confidence')]
for (strategy1, strategy2) in strategy_pairs:
    index1, index2 = strategies.index(strategy1), strategies.index(strategy2)
    """calculate_and_print_overlap_statistics(aggregated_data_per_strategy[index1], 
                                           aggregated_data_per_strategy[index2], 
                                           f"{strategy1} vs. {strategy2}")"""
    calculate_and_print_overlap_statistics_relative_to_energy(aggregated_data_per_strategy[index1],
                                                              aggregated_data_per_strategy[index2],
                                                              f"{strategy1} vs. {strategy2}")


