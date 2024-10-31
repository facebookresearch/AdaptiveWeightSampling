# AdaptiveWeightSampling

## About

We consider a range of loss-based sampling strategies and smooth convex training loss functions and propose a novel active learning algorithm called *Adaptive-Weight Sampling (AWS)* that utilizes SGD with an adaptive step size that achieves stochastic Polyak’s step size in expectation. In our paper, we establish convergence rate results for AWS for smooth convex training loss functions. In this repository, contains the implementation of AWS along with code for numerical experiments that demonstrate the efficiency of AWS on various datasets by using either exact or estimated loss values.

This repository contains sources codes for the following paper:

> [Haimovich, D., Karamshuk, D., Linder, F., Tax, N., & Vojnovic, M. On the convergence of loss and uncertainty-based active learning algorithms](https://arxiv.org/pdf/2312.13927). *Advances in Neural Information Processing Systems* 37 (2024).

The Jupyter notebooks in this repository reproduce the experiments presented in the paper. Implementation of AdaptiveWeightSampling can be found in the *model.py* file.

## Instructions
The `experiments` folder contains notebooks to repeat the experimental setup described in the paper and produce the figures from the paper. This folder contains one notebook for each of the dataset, with the exception of the mushroom dataset (see note below).

The AdaptiveWeightSampling method itself is implemented in `src/models.py`, and more generally the `src` folder contains various utility functions that are required to run these notebooks.

To run these experiments notebooks:

    - Navigate to the root of this repository
    - Run `pip install .` to install the code in `src` as a python package. It is recommended to run this in a virtual environment, e.g. using `virtualenv`. This is required to make all the import statements in the notebooks work.
    - Run the notebook.

### Note on the Mushrooms dataset
The variant of the mushrooms dataset that we used in our experimental setup contains features that were extracted using an RBF-kernel, introduced in:

> Loizou, N., Vaswani, S., Laradji, I. H., & Lacoste-Julien, S. (2021, March). Stochastic polyak step-size for sgd: An adaptive learning rate for fast convergence. In International Conference on Artificial Intelligence and Statistics (pp. 1306-1314). PMLR.

Users can manually obtain this dataset can be using the code from this paper's code repository: https://github.com/IssamLaradji/sps.

## Contributing
See CONTRIBUTING.md

## Code of conduct
See CODE_OF_CONDUCT.md

## License

The majority of AdaptiveWeightSampling is licensed under CC-BY-NC, however portions of the project are available under separate license terms: TensorFlow is licensed under the Apache 2.0 license, NumPy, Pandas, Scikit-learn, and Seaborn are licensed under the BSD-3 license, Matplotlib is licensed under the PSF license, and Ucimlrepo is licensed under the MIT license. See LICENCE.md for details.
