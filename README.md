# Section 3: *Towards Objective Identifiers of Hard Samples*

## Description

The [replicate_inversion_results_on_multiple_classes.py](replicate_inversion_results_on_multiple_classes.py) program 
generalizes the approach introduced by [Ciceri et al. (2024)](https://www.nature.com/articles/s42256-023-00772-9) to
multiclass classification. The program trains 5 models with different initialization and computes the evolution of the 
radii of class manifolds over epochs, plotting the results for visualization. The results prove the generalizability of
the 'inversion point' to multiclass classification, albeit with an emergence of class-specific characteristics.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- tqdm
- numpy

Make sure you have the required libraries installed. You can install them using pip:

```bash
pip install torch torchvision matplotlib tqdm numpy
```

## Running the Code

To run [replicate_inversion_results_on_multiple_classes.py](replicate_inversion_results_on_multiple_classes.py), use
the following command in your terminal:

```bash
python replicate_inversion_results_on_multiple_classes.py --dataset_name [DATASET_NAME] --subset_size [SUBSET_SIZE]
```

### Parameters

- `--dataset_name`: Specifies the name of the dataset to load for the analysis. The dataset must be available in 
`torchvision.datasets`. Currently, the code only works for single-colour datasets. **Default value is `MNIST`.**
- `--subset_size`: Integer specifying the size of the dataset subset to use for training and analysis. This allows for 
quicker iterations during experimentation. **Default value is `20000`.** Actual experiments were conducted on `70000`.

## Expected Results

When running `python replicate_inversion_results_on_multiple_classes.py --dataset_name MNIST --subset_size 20000`, the 
following result is observed:

![Radii Evolution Over Epochs on MNIST](Figures/radii_on_MNIST.png)

The results will differ each time due to the random initialization, however the existence of the inversion point should
be still verifiable, even when the `--subset_size` is reduced to `20000`.

# Sections 4-5: In-class Data Imbalance & Benchmarking Hard Sample Identification Methods

## Description

The [verifying_importance_of_stragglers.py](verifying_importance_of_stragglers.py) program allows for measuring the
performance of a SimpleNN on the easy and hard samples from the dataset specified by the user. This allows to discover
the in-class imbalance phenomenon, and also allows for comparison between the degree of the observed in-class imbalance
depending on the method used for identifying hard samples.

## Requirements

The [verifying_importance_of_stragglers.py](verifying_importance_of_stragglers.py) has the same requirements as the ones 
described [here](#requirements).