import argparse
import pickle

import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import initialize_model, load_data_and_normalize, plot_radii, train_model, transform_datasets_to_dataloaders


def main(dataset_name: str, subset_size: int):
    dataset = load_data_and_normalize(dataset_name, subset_size)
    loader = transform_datasets_to_dataloaders(dataset)
    all_epoch_radii = []
    for _ in tqdm(range(5), desc='Investigating the dynamics of the radii of class manifolds for distinctly initialized'
                                 ' networks'):
        model, optimizer = initialize_model()
        epoch_radii = train_model(model, loader, optimizer)
        all_epoch_radii.append(epoch_radii)
    # Save and plot the results
    with open('Results/all_epoch_radii.pkl', 'wb') as f:
        pickle.dump(all_epoch_radii, f)
    plot_radii('Epoch', all_epoch_radii, dataset_name, True)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Investigate the dynamics of the radii of class manifolds for distinctly initialized networks.')
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        help='Name of the dataset to load. It has to be available in torchvision.datasets')
    parser.add_argument('--subset_size', type=int, default=20000,
                        help='Size of the subset to use for the analysis.')
    args = parser.parse_args()
    main(args.dataset_name, args.subset_size)
