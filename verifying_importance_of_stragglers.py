import argparse
from typing import List

import numpy as np
import tqdm

from utils import load_data_and_normalize, identify_hard_samples, plot_generalisation, straggler_ratio_vs_generalisation


def main(dataset_name: str, strategy: str, train_ratios: List[float], remaining_train_ratios: List[float],
         reduce_hard: bool, level: str, noise_ratio: float, subset_size: int):
    dataset = load_data_and_normalize(dataset_name, subset_size)
    generalisation_settings = ['full', 'hard', 'easy']
    total_avg_accuracies = {setting: {ratio: [] for ratio in train_ratios} for setting in generalisation_settings}
    total_std_accuracies = {setting: {ratio: [] for ratio in train_ratios} for setting in generalisation_settings}
    for idx, train_ratio in tqdm.tqdm(enumerate(train_ratios), desc='Going through different train:test ratios'):
        current_accuracies = {setting: {reduce_train_ratio: [] for reduce_train_ratio in remaining_train_ratios}
                              for setting in generalisation_settings}
        for _ in tqdm.tqdm(range(2), desc='Repeating the experiment for different straggler sets'):
            hard_data, hard_target, easy_data, easy_target = identify_hard_samples(strategy, dataset, level,
                                                                                   noise_ratio)
            print(f'A total of {len(hard_data)} hard samples and {len(easy_data)} easy samples were found.')
            straggler_ratio_vs_generalisation(hard_data, hard_target, easy_data, easy_target, train_ratio,
                                              reduce_hard, remaining_train_ratios, current_accuracies)
        # Compute the average and standard deviation of accuracies for each ratio
        avg_accuracies = {generalisation_settings[i]:
                          [np.mean(current_accuracies[generalisation_settings[i]][remaining_train_ratio])
                           for remaining_train_ratio in remaining_train_ratios] for i in range(3)}
        std_accuracies = {generalisation_settings[i]:
                          [np.std(current_accuracies[generalisation_settings[i]][remaining_train_ratio])
                           for remaining_train_ratio in remaining_train_ratios] for i in range(3)}
        print(f'For train_ratio = {train_ratio} we get average accuracies of {avg_accuracies["full"]}% on full test set'
              f', {avg_accuracies["hard"]}% on hard test samples and {avg_accuracies["easy"]}% on easy test samples.')
        for setting in generalisation_settings:
            total_avg_accuracies[setting][train_ratio] = avg_accuracies[setting]
            total_std_accuracies[setting][train_ratio] = std_accuracies[setting]

    plot_generalisation(train_ratios, remaining_train_ratios, reduce_hard, total_avg_accuracies, strategy, level,
                        total_std_accuracies, noise_ratio, dataset_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script to investigate the impact of reducing hard/easy samples on generalisation.')
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        help='Dataset name. The code was tested on MNIST, FashionMNIST and KMNIST.')
    parser.add_argument('--strategy', type=str, choices=['stragglers', 'confidence', 'energy'],
                        default='stragglers', help='Strategy (method) to use for identifying hard samples.')
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
    parser.add_argument('--subset_size', type=int)

    args = parser.parse_args()
    main(**vars(args))
