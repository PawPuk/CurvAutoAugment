import torch

from utils import load_data, identify_hard_samples, straggler_ratio_vs_generalisation

# Load the dataset. Work on full batch.
train_dataset, test_dataset, full_dataset = load_data()
full_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True)

generalisation_settings = ['full', 'stragglers', 'non_stragglers']
stragglers_data, stragglers_target, non_stragglers_data, non_stragglers_target = identify_hard_samples(full_loader)
straggler_ratio_vs_generalisation(0.9, stragglers_data, stragglers_target, non_stragglers_data,
                                  non_stragglers_target, 0.99, True, full_loader)
