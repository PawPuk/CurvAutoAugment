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

## Running the Code

To run [verifying_importance_of_stragglers.py](verifying_importance_of_stragglers.py), use
the following command in your terminal:

```bash
python verifying_importance_of_stragglers.py --dataset_name [DATASET_NAME] --strategy [STRATEGY] --train_ratios [TRAIN_RATIOS] --remaining_train_ratios [REMAINING_TRAIN_RATIOS] --reduce_hard [REDUCE_HARD] --level [LEVEL] --noise_ratio [NOISE_RATIO] --subset_size [SUBSET_SIZE]
```

## Parameters

- `--dataset_name`: Specifies the name of the dataset to load for the analysis. The dataset must be available in 
`torchvision.datasets`. The code was tested on `MNIST`, `FashionMNIST`, and `KMNIST`. **Default value is `MNIST`.**

- `--strategy`: Specifies the strategy (method) to use for identifying hard samples. The possible options are 
`stragglers`, `confidence`, and `energy`. **Default value is `stragglers`.**

- `--train_ratios`: A list of percentages representing the train set to the whole dataset size, used to infer 
training:test ratio. For example, `0.9 0.5` means training with 90% and 50% of the data, respectively. 
**Default values are `[0.9, 0.5]`.** Actual experiments were conducted on `[0.9, 0.8, 0.7, 0.6, 0.5]`.

- `--remaining_train_ratios`: A list of percentages of train hard/easy samples on which we train; we only reduce the 
number of hard OR easy samples (depending on the `--reduce_hard` flag). So, `0.1` means that 90% of hard samples will 
be removed from the train set before training (when `reduce_hard == True`). **Default values are 
`[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]`.** Actual experiments were conducted on 
`[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]` when `reduce_hard == False`, and `[0.0, 0.25, 0.5, 0.75, 1.0]` otherwise.

- `--reduce_hard`: A flag indicating whether to reduce the number of hard (True) or easy (False) samples from the 
training data. **By default, this is set to `False`, focusing on reducing easy samples.**

- `--level`: Specifies the level at which confidence and energy are computed, with possible options being `class` and 
`dataset`. **Default value is `dataset`.** This choice affects how the analysis interprets individual sample difficulty 
or energy levels across the entire dataset or within each class.

- `--noise_ratio`: The ratio of the added label noise. Setting this will introduce noisy labels into the dataset, with 
`(100*noise_ratio)%` of labels becoming incorrect. This simulates a dataset with existing labeling errors to examine 
the model's robustness to label noise. **Default value is `0.0`, meaning no noise is added by default.**

- `--subset_size`: Integer specifying the size of the dataset subset to use for training and analysis. This parameter 
allows for quicker iterations during experimentation. **There is no default value provided in the code snippet; ensure 
to specify this parameter when running the script.** Actual experiments were conducted on subsets up to `70000`, but 
using `20000` should give good enough results (although less reliable).
