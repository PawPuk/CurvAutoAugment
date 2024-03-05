import matplotlib.pyplot as plt
import numpy as np
import tqdm

from utils import (load_data_and_normalize, identify_hard_samples, straggler_ratio_vs_generalisation,
                   transform_datasets_to_dataloaders)

train_ratios = [0.9, 0.8, 0.7, 0.6, 0.5]  # train:test ratio (percentage of train set to whole dataset)
strategy = "confidence"  # choose from "stragglers", "confidence", "energy"
dataset_name = 'MNIST'
reduce_train_ratios = [0.0, 0.05, 0.1, 0.15, 0.2]  # removed train hard/easy samples (%)
reduce_hard = False  # True/False - see the impact of reducing hard/easy samples on generalisation
level = 'dataset'  # Choose between 'class' and 'dataset' (only used when strategy is 'softmax' or 'energy')
noise_ratio = 0.0  # Has to be in [0, 1)

# Load the dataset. Use full batch. When not using straggler-based approach the noise is added later (passing 0.0 now)
dataset = load_data_and_normalize(dataset_name, 20000, [0.0, noise_ratio][strategy == 'stragglers'])
loader = transform_datasets_to_dataloaders(dataset)
# Define the color scheme and initialize necessary variables
colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(train_ratios)))  # Darker to lighter blues
generalisation_settings = ['full', 'hard', 'easy']
total_avg_accuracies = {setting: {ratio: [] for ratio in train_ratios} for setting in generalisation_settings}
total_std_accuracies = {setting: {ratio: [] for ratio in train_ratios} for setting in generalisation_settings}

for idx, train_ratio in tqdm.tqdm(enumerate(train_ratios), desc='Going through different train:test ratios'):
    current_accuracies = {setting: {reduce_train_ratio: [] for reduce_train_ratio in reduce_train_ratios}
                          for setting in generalisation_settings}
    for run_index in tqdm.tqdm(range(1), desc='Repeating the experiment for different straggler sets'):
        hard_data, hard_target, easy_data, easy_target = identify_hard_samples(strategy, loader, dataset, level,
                                                                               noise_ratio)
        print(f'A total of {len(hard_data)} hard samples and {len(easy_data)} easy samples were found.')
        straggler_ratio_vs_generalisation(reduce_train_ratios, hard_data, hard_target, easy_data, easy_target,
                                          train_ratio, reduce_hard, current_accuracies)
    print(current_accuracies)
    # Compute the average and standard deviation of accuracies for each ratio
    avg_accuracies = {generalisation_settings[i]:
                      [np.mean(current_accuracies[generalisation_settings[i]][reduce_train_ratio])
                       for reduce_train_ratio in reduce_train_ratios] for i in range(3)}
    std_accuracies = {generalisation_settings[i]:
                      [np.std(current_accuracies[generalisation_settings[i]][reduce_train_ratio])
                       for reduce_train_ratio in reduce_train_ratios] for i in range(3)}
    print(f'For train_ratio = {train_ratio} we get average accuracies of {avg_accuracies["full"]}% on full test set'
          f', {avg_accuracies["stragglers"]}% on test stragglers and {avg_accuracies["non_stragglers"]}% on '
          f'test non-stragglers.')
    for setting in generalisation_settings:
        total_avg_accuracies[setting][train_ratio] = avg_accuracies[setting]
        total_std_accuracies[setting][train_ratio] = std_accuracies[setting]

for setting in generalisation_settings:
    for idx in range(len(train_ratios)):
        ratio = train_ratios[idx]
        plt.errorbar(reduce_train_ratios, total_avg_accuracies[setting][ratio],
                     yerr=total_std_accuracies[setting][ratio],
                     label=f'Train:Test={int(100*ratio)}:{100-int(100*ratio)}', marker='o', markersize=5, capsize=5,
                     linewidth=2, color=colors[idx])
    plt.xlabel(f'Proportion of {["Non-Stragglers", "Stragglers"][reduce_hard]} Removed from Train Set',
               fontsize=14)
    plt.ylabel(f'Accuracy on {setting.capitalize()} Test Set (%)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.legend(title='Train:Test Ratio')
    plt.tight_layout()  # Adjust the padding between and around subplots.
    s = ''
    if strategy != 'stragglers':
        s = f'{level}_'
    plt.savefig(f'Figures/generalisation_from_{["non_stragglers", "stragglers"][reduce_hard]}_to_{setting}_on_'
                f'{dataset_name}_using_{s}{strategy}_{noise_ratio}noise.png')
    plt.savefig(f'Figures/generalisation_from_{["non_stragglers", "stragglers"][reduce_hard]}_to_{setting}_on_'
                f'{dataset_name}_using_{s}{strategy}_{noise_ratio}noise.pdf')
    plt.clf()
