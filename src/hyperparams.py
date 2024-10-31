# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from sklearn.ensemble import RandomForestRegressor

BY_DATASET = {
    "parkinsons": {
        "polyak_absloss": {"kappa": 5, "kappa0": 0.9, "eta": 0.01, "omega": 1.0},
        "polyak_absloss_lr": {
            "kappa": 5,
            "kappa0": 0.9,
            "eta": 0.01,
            "omega": 1.0,
            "warmup_steps_estimator": 5,
            "absloss_estimator": RandomForestRegressor(n_estimators=100),
        },
        "absloss": {"eta": 0.01},
        "absloss_lr": {
            "eta": 0.01,
            "warmup_steps_estimator": 5,
            "absloss_estimator": RandomForestRegressor(n_estimators=100),
        },
        "random": {"eta": 0.01},
    },
    "tictactoe": {
        "polyak_absloss": {"kappa": 1.5, "kappa0": 0.99, "eta": 0.05, "omega": 1.0},
        "polyak_absloss_lr": {
            "kappa": 1.5,
            "kappa0": 0.99,
            "eta": 0.05,
            "omega": 1.0,
            "warmup_steps_estimator": 1,
            "absloss_estimator": RandomForestRegressor(n_estimators=100),
        },
        "absloss": {"eta": 0.05},
        "absloss_lr": {
            "eta": 0.05,
            "warmup_steps_estimator": 1,
            "absloss_estimator": RandomForestRegressor(n_estimators=100),
        },
        "random": {"eta": 0.05},
    },
    "mushroom": {
        "polyak_absloss": {
            "kappa": 35.52674391886635,
            "kappa0": 0.998009193683614,
            "eta": 300,
            "omega": 0.5588345498272365,
        },
        "polyak_absloss_lr": {
            "kappa": 35.52674391886635,
            "kappa0": 0.998009193683614,
            "eta": 300,
            "omega": 0.5588345498272365,
            "warmup_steps_estimator": 1,
            "absloss_estimator": RandomForestRegressor(n_estimators=25),
        },
        "absloss": {"eta": 300},
        "absloss_lr": {
            "eta": 300,
            "warmup_steps_estimator": 1,
            "absloss_estimator": RandomForestRegressor(n_estimators=25),
        },
        "random": {"eta": 300},
    },
    "mnist": {
        "polyak_absloss": {"kappa": 10, "kappa0": 0.99, "eta": 0.1, "omega": 1.1},
        "polyak_absloss_lr": {
            "kappa": 10,
            "kappa0": 0.99,
            "eta": 0.1,
            "omega": 1.1,
            "warmup_steps_estimator": 50,
            "absloss_estimator": RandomForestRegressor(n_estimators=25),
        },
        "absloss": {"eta": 0.1},
        "absloss_lr": {
            "eta": 0.1,
            "warmup_steps_estimator": 50,
            "absloss_estimator": RandomForestRegressor(n_estimators=25),
        },
        "random": {"eta": 0.1},
    },
    "credit": {
        "polyak_absloss": {"kappa": 10, "kappa0": 0.99, "eta": 0.1, "omega": 1.1},
        "polyak_absloss_lr": {
            "kappa": 10,
            "kappa0": 0.99,
            "eta": 0.1,
            "omega": 1.1,
            "warmup_steps_estimator": 50,
            "absloss_estimator": RandomForestRegressor(n_estimators=100),
        },
        "absloss": {"eta": 0.1},
        "absloss_lr": {
            "eta": 0.1,
            "warmup_steps_estimator": 50,
            "absloss_estimator": RandomForestRegressor(n_estimators=100),
        },
        "random": {"eta": 0.1},
    },
    "splice": {
        "polyak_absloss": {"kappa": 10, "kappa0": 0.9, "eta": 0.1, "omega": 1.0},
        "polyak_absloss_lr": {
            "kappa": 10,
            "kappa0": 0.9,
            "eta": 0.1,
            "omega": 1.0,
            "warmup_steps_estimator": 25,
            "absloss_estimator": RandomForestRegressor(n_estimators=100),
        },
        "absloss": {"eta": 0.1},
        "absloss_lr": {
            "eta": 0.1,
            "warmup_steps_estimator": 25,
            "absloss_estimator": RandomForestRegressor(n_estimators=100),
        },
        "random": {"eta": 0.1},
    },
}
