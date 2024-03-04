# Section 3: *Towards Objective Identifiers of Hard Samples*

## Description

The [replicate_inversion_results_on_multiple_classes.py](./replicate_inversion_results_on_multiple_classes.py) program 
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

To run [replicate_inversion_results_on_multiple_classes.py](./replicate_inversion_results_on_multiple_classes.py), use
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

![Radii Evolution Over Epochs on MNIST](./Figures/radii_over_Epoch_on_MNIST.png)

The results will differ each time due to the random initialization, however the existence of the inversion point should
be still verifiable, even when the `--subset_size` is reduced to `20000`.