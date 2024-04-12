import pickle
import glob
import matplotlib.pyplot as plt
import numpy as np


def plot_combined_metrics(file_path, figs, axs, idx_offset):
    # Determine color and marker based on the file name
    if 'energy' in file_path:
        color = 'blue'
        marker = 'o'  # Actual marker used for energy
        label_prefix = 'Energy-based'
    elif 'confidence' in file_path:
        color = 'green'
        marker = 's'  # Actual marker used for confidence
        label_prefix = 'Confidence-based'
    elif 'stragglers' in file_path:
        color = 'red'
        marker = 'x'  # Actual marker used for stragglers
        label_prefix = 'Straggler-based'
    else:
        color = 'black'  # Default color if none of the above matches
        marker = 'd'  # Default marker
        label_prefix = 'Unknown-based'

    with open(file_path, 'rb') as f:
        all_metrics = pickle.load(f)

    # Define the metric names and settings
    metrics_names = ['accuracy', 'precision', 'recall', 'f1']
    settings = ['hard', 'easy']
    line_styles = {'hard': 'solid', 'easy': 'dotted'}

    # Extract ratios and sort them for consistent plotting
    ratios = sorted(next(iter(all_metrics.values()))[list(next(iter(all_metrics.values())).keys())[0]].keys())
    ratios_float = [float(ratio) for ratio in ratios]

    for idx, metric_name in enumerate(metrics_names):
        for setting in settings:
            mean_values = []
            std_values = []

            # Extract data and compute mean and std deviation for each ratio
            for ratio in ratios:
                if ratio in all_metrics[next(iter(all_metrics))][setting]:
                    metric_values = [all_metrics[tr][setting][ratio][metric_name] for tr in all_metrics]
                    metric_values_flat = [val for sublist in metric_values for val in sublist]
                    mean_values.append(np.mean(metric_values_flat))
                    std_values.append(np.std(metric_values_flat))
                else:
                    mean_values.append(np.nan)
                    std_values.append(np.nan)

            axs[idx].plot(
                ratios_float, mean_values, linestyle=line_styles[setting], marker=marker, color=color,
                label=f'{label_prefix} {setting.capitalize()} {metric_name.capitalize()}'
            )
            axs[idx].fill_between(
                ratios_float, np.subtract(mean_values, std_values), np.add(mean_values, std_values),
                color=color, alpha=0.2
            )

        axs[idx].set_title(metric_name.capitalize())
        axs[idx].set_xlabel(f'Proportion of {["Hard", "Easy"][easy]} Samples Remaining in Training Set')
        axs[idx].set_ylabel(metric_name.capitalize())
        axs[idx].set_xticks(ratios_float)
        axs[idx].grid(True)


easy = False
# Prepare figures and axes for each metric
figs, axs = [], []
metrics_names = ['accuracy', 'precision', 'recall', 'f1']
for i in range(4):  # 4 metrics
    fig, ax = plt.subplots(figsize=(11, 7))
    figs.append(fig)
    axs.append(ax)
    custom_lines = [
        plt.Line2D([0], [0], color='black', linestyle='solid', lw=2, label='Hard samples'),
        plt.Line2D([0], [0], color='black', linestyle='dotted', lw=2, label='Easy samples')
    ]
    line_marker_legend = plt.legend(handles=custom_lines, title='Accuracy on:', loc='center left',
                                    bbox_to_anchor=(0.7, 0.8))
    plt.gca().add_artist(line_marker_legend)
    method_legend_labels = [
        plt.Line2D([0], [0], color='blue', marker='o', lw=0, markersize=10, label='Energy-based'),
        plt.Line2D([0], [0], color='green', marker='s', lw=0, markersize=10, label='Confidence-based'),
        plt.Line2D([0], [0], color='red', marker='x', lw=0, markersize=10, label='Straggler-based')
    ]
    plt.legend(handles=method_legend_labels, title='Identifying Hard Samples with:', loc='lower center',
               bbox_to_anchor=(0.8, 0.53))

files = glob.glob(f'Results/F1_results/MNIST_*_{["True", "False"][easy]}_20000_metrics.pkl')

# Iterate over files and plot metrics on each figure
for file_path in files:
    plot_combined_metrics(file_path, figs, axs, 0)  # idx_offset is not used here, can be removed
names = [f'Figures/Figure 7/{["e) hard", "a) easy"][easy]}_accuracy.pdf',
         f'Figures/Figure 7/{["f) hard", "b) easy"][easy]}_recall.pdf',
         f'Figures/Figure 7/{["g) hard", "c) easy"][easy]}_precision.pdf',
         f'Figures/Figure 7/{["h) hard", "d) easy"][easy]}_F1.pdf']
for i, fig in enumerate(figs):
    fig.savefig(names[i])
plt.show()
