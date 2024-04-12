import argparse
import pickle

import torch
from tqdm import tqdm

from utils import initialize_model, load_data_and_normalize, plot_radii, train_model, transform_datasets_to_dataloaders


def main(dataset_name: str, subset_size: int, network: str):
    dataset = load_data_and_normalize(dataset_name, subset_size)
    loader = transform_datasets_to_dataloaders(dataset)
    all_epoch_radii = []
    for _ in tqdm(range(3), desc='Investigating the dynamics of the radii of class manifolds for distinctly initialized'
                                 ' networks'):
        # Define Model, Loss, and Optimizer
        model, optimizer = initialize_model(evaluation_network=network, dataset=dataset_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = torch.nn.DataParallel(model)
        model.to(device)
        epoch_radii = train_model(model, loader, optimizer)
        all_epoch_radii.append(epoch_radii)
    with open(f'Results/all_epoch_radii_{subset_size}{dataset_name}.pkl', 'wb') as f:
        pickle.dump(all_epoch_radii, f)
    plot_radii(all_epoch_radii, 'MNIST', True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Investigate the dynamics of the radii of class manifolds for distinctly initialized networks.')
    parser.add_argument('--dataset_name', type=str, default='MNIST',
                        help='Name of the dataset to load. It has to be available in torchvision.datasets.')
    parser.add_argument('--subset_size', type=int, default=20000,
                        help='Size of the subset to use for the analysis.')
    parser.add_argument('--network', type=str, default='SimpleNN',
                        help='The network that will be used to find the inversion points.')
    args = parser.parse_args()
    main(args.dataset_name, args.subset_size, args.network)
