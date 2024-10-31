# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from setuptools import find_packages, setup

setup(
    name="adaptive_weight_sampling",
    version="1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy==1.26.4",
        "pandas==2.2.3",
        "scikit-learn==1.5.2",
        "ucimlrepo==0.0.7",
        "seaborn==0.12.2",
        "matplotlib==3.8.4",
    ],
)
