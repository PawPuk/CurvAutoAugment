import argparse
import pickle
from typing import List

import torch
import tqdm

from utils import load_data_and_normalize, identify_hard_samples, straggler_ratio_vs_generalisation


def main(dataset_name: str, strategy: str, runs: int, train_ratios: List[float],
         remaining_train_ratios: List[float], reduce_hard: bool, level: str, noise_ratio: float, subset_size: int,
         evaluation_network: str):
    dataset = load_data_and_normalize(dataset_name, subset_size)
    generalisation_settings = ['full', 'hard', 'easy']
    indices_of_hard_samples = []
    all_metrics = {}
    for idx, train_ratio in tqdm.tqdm(enumerate(train_ratios), desc='Going through different train:test ratios'):
        current_metrics = {setting: {reduce_train_ratio: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
                                     for reduce_train_ratio in remaining_train_ratios}
                           for setting in generalisation_settings}
        for _ in tqdm.tqdm(range(runs), desc='Repeating the experiment for different straggler sets'):
            hard_data, hard_target, easy_data, easy_target, hard_indices = identify_hard_samples(strategy, dataset,
                                                                                                 level, noise_ratio)
            print(f'A total of {len(hard_data)} hard samples and {len(easy_data)} easy samples were found.')
            straggler_ratio_vs_generalisation(hard_data, hard_target, easy_data, easy_target, train_ratio, reduce_hard,
                                              remaining_train_ratios, current_metrics, evaluation_network)
            hard_indices = [h_index.cpu() for h_index in hard_indices] if torch.cuda.is_available() else hard_indices
            indices_of_hard_samples.append(hard_indices)
        # After each train_ratio, add the collected metrics to the all_metrics dictionary
        all_metrics[train_ratio] = current_metrics
    metrics_filename = f"{dataset_name}_{strategy}_{reduce_hard}_{subset_size}_metrics.pkl"
    indices_filename = f"{dataset_name}_{strategy}_{reduce_hard}_{subset_size}_indices.pkl"
    with open(metrics_filename, 'wb') as f:
        pickle.dump(all_metrics, f)
    with open(indices_filename, 'wb') as f:
        pickle.dump(indices_of_hard_samples, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script to investigate the impact of reducing hard/easy samples on generalisation.')
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        help='Dataset name. The code was tested on MNIST, FashionMNIST and KMNIST.')
    parser.add_argument('--strategy', type=str, choices=['stragglers', 'confidence', 'energy'],
                        default='stragglers', help='Strategy (method) to use for identifying hard samples.')
    parser.add_argument('--runs', type=int, default=3,
                        help='Specifies how many straggler sets will be computed for the experiment, and how many '
                             'networks will be trained per a straggler set (for every ratio in remaining_train_ratios. '
                             'The larger this value the higher the complexity and the statistical significance.')
    parser.add_argument('--train_ratios', nargs='+', type=float, default=[0.9, 0.5],
                        help='Percentage of train set to whole dataset - used to infer training:test ratio.')
    parser.add_argument('--remaining_train_ratios', nargs='+', type=float,
                        default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0],
                        help='Percentage of train hard/easy samples on which we train; we only reduce the number of '
                             'hard OR easy samples (depending on --reduce_hard flag). So 0.1 means that 90% of hard '
                             'samples will be removed from the train set before training (when reduce_hard == True).')
    parser.add_argument('--reduce_hard', action='store_true', default=False,
                        help='flag indicating whether we want to see the effect of changing the number of easy (False) '
                             'or hard (True) samples.')
    parser.add_argument('--level', type=str, choices=['class', 'dataset'], default='dataset',
                        help='Specifies the level at which the energy is computed. Is also affects how the hard samples'
                             ' are chosen in confidence- and energy-based methods')
    parser.add_argument('--noise_ratio', type=float, default=0.0,
                        help='The ratio of the added label noise. After this, the dataset will contain (100*noise_ratio'
                             ')% noisy-labels (assuming all labels were correct prior to calling this function).')
    parser.add_argument('--subset_size', default=20000, type=int,
                        help='Specifies the subset of the dataset used for the experiments. Later it will be divided '
                             'into train and testing training and test sets based pm the --train_ratios parameter.')
    parser.add_argument('--evaluation_network', default='SimpleNN', choices=['SimpleNN', 'ResNet'],
                        help='Specifies the network that will be used for evaluating the performance on hard and easy '
                             'data. This shows that no matter the network used, the hard samples remain hard to learn')
    args = parser.parse_args()
    main(**vars(args))
