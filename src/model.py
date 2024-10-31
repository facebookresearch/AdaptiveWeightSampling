# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import math

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError

from utils import (
    accuracy_prob,
    backward_logloss,
    cross_entropy,
    forward_logloss,
    log_ratio,
    ratio,
    update,
)

loss = cross_entropy
forward = forward_logloss
backward = backward_logloss
accuracy = accuracy_prob


def sample_train_evaluate_loop(
    X: pd.DataFrame,
    y: pd.DataFrame,
    Xtest: pd.DataFrame,
    ytest: pd.DataFrame,
    mode="rand",
    eta=0.01,
    lam=1,
    # Used in all the Polyak methods
    kappa=1,
    kappa0=0.1,
    # Used in Polyak and Polyak-bec
    polyakexp=1 / 2,
    # Used in polyak-becpi
    omega=1,
    # Added to enable non-adaptive stepsize experiments with pz != 1/2
    pz0=1 / 2,
    # For bec-minx1
    alpha=1,
    x0=1,
    # multiplier to control sampling probability in bec-absloss
    prob_const=1,
    # debug
    verbose=True,
    epsilon=1e-45,
    max_iterations=20000,
    # parameters of the noisy estimators
    noise_beta_a=1,
    absloss_estimator=RandomForestRegressor(),
    warmup_steps_estimator=1,
):
    theta = np.zeros(X.shape[1])

    n_iterations = min(X.shape[0], max_iterations)  # Number of iterations

    if verbose:
        print(
            f"Training with algorithm {mode}: {n_iterations} iterations with d={X.shape[1]}"
        )

    losses = np.zeros(n_iterations)
    losses_reg = np.zeros(n_iterations)
    losses_test = np.zeros(n_iterations)
    losses_test_reg = np.zeros(n_iterations)
    accuracies = np.zeros(n_iterations)
    accuracies_test = np.zeros(n_iterations)
    labeled = np.zeros(n_iterations)
    probs = np.zeros(n_iterations)
    # for noisy estimator
    absloss_gt = np.zeros(n_iterations)
    absloss_est = np.zeros(n_iterations)
    absloss_X_train = []
    absloss_y_train = []
    absloss_yy_train = []
    absloss_xx_train = []
    pz_gt = None
    xx_ext = None

    for i in range(n_iterations):
        xx = X[i, :]
        yy = y[i]

        p = 0
        pz = 1 / 2
        grad = 0

        if mode == "random":
            pz = pz0
            p = forward(xx, theta)
            grad = backward(xx, yy, p)
        elif mode == "polyak_absloss":
            p = forward(xx, theta)
            grad = backward(xx, yy, p)
            u = (2 * yy - 1) * np.dot(xx, theta)
            zeta = kappa * ratio(u) / (np.linalg.norm(xx) ** 2)
            zeta = min(zeta, kappa0)
            pz = omega / (1 + np.exp(u))
            pz = np.clip(pz, a_min=0, a_max=1)
            grad = zeta * grad / pz
        elif mode == "polyak_random":
            """
            Added to address the reviewer's comment:
            The authors conduct experiments on "uniform sampling" + "constant step size", "loss-based sampling" + "constant step size",
            and "loss-based sampling" + "Polyak step size" to verify the effectiveness of the approach of loss-based sampling. For
            completeness, it is necessary to present the performance of using "uniform sampling" + "Polyak step size" in the numerical experiments.
            """
            p = forward(xx, theta)
            grad = backward(xx, yy, p)
            u = (2 * yy - 1) * np.dot(xx, theta)
            zeta = kappa * ratio(u) / (np.linalg.norm(xx) ** 2)
            zeta = min(zeta, kappa0)
            pz = pz0
            grad = zeta * grad / pz
        elif mode == "polyak_exponent":
            p = forward(xx, theta)
            grad = backward(xx, yy, p)
            u = (2 * yy - 1) * np.dot(xx, theta)
            zeta_log = (
                math.log(kappa) + log_ratio(u) - math.log(np.linalg.norm(xx) ** 2)
            )
            zeta = math.exp(min(zeta_log, math.log(kappa0)))
            pz = zeta**polyakexp
            pz = np.clip(pz, a_min=epsilon, a_max=1 - epsilon)
            grad = (zeta ** (1 - polyakexp)) * grad
        elif mode == "polyak_exponent_old":
            p = forward(xx, theta)
            grad = backward(xx, yy, p)
            u = (2 * yy - 1) * np.dot(xx, theta)
            zeta = kappa * ratio(u) / (np.linalg.norm(xx) ** 2)
            zeta = min(zeta, kappa0)
            pz = zeta**polyakexp
            grad = (zeta ** (1 - polyakexp)) * grad
        elif mode == "absloss":
            p = forward(xx, theta)
            bec = cross_entropy(p, yy)
            pz = 1 - np.exp(-bec)
            pz = omega * pz
            pz = np.clip(pz, a_min=epsilon, a_max=1 - epsilon)
            grad = backward(xx, yy, p)
        elif mode == "minx1":
            p = forward(xx, theta)
            bec = cross_entropy(p, yy)
            pz = np.min([(bec / x0) ** alpha, 1])
            grad = backward(xx, yy, p)
        elif mode == "polyak-minx1":
            # Binary cross-entropy loss with Polyak adaptive step size
            p = forward(xx, theta)
            bec = cross_entropy(p, yy)
            grad = backward(xx, yy, p)

            # Compute pz as in bec-minx1 mode
            pz = np.min([(bec / x0) ** alpha, 1])

            # Compute u and zeta as in polyak-bec mode
            u = (2 * yy - 1) * np.dot(xx, theta)
            zeta = kappa * ratio(u) / (np.linalg.norm(xx) ** 2)
            zeta = min(zeta, kappa0)

            # Apply the Polyak adaptive step size to the gradient
            scaling_factor = zeta ** (1 - polyakexp)
            grad = scaling_factor * grad
        elif mode == "absloss-betanoise":
            # We want to model a noisy loss estimator \hat(pz) of pz.
            #
            # To this end we want to choose a distribution f(\hat(pz)) of random variable \hat(pz),
            # such that \hat(pz) \in [0, 1] (i.e. f(\hat(pz)) = 0 for \hat(pz) not in [0, 1]) and
            # E[\hat(pz)] = pz and f(\hat(pz)) has only one mode.
            #
            # f(\hat(pz)) = beta(a, b) where a is a hyper-parameter, b(\hat(pz)) = a * (1 - pz) / pz
            # and ((a > 1) or (b > 1)).
            #
            # The variance can be estimated as var(\hat(pz)) = (pz^2 - pz^3) / (a + pz) and depends on
            # the mean and parameter a. Here is some visualization of var(\hat(pz)) https://www.desmos.com/calculator/u0oodh6msb.
            p = forward(xx, theta)
            bec = cross_entropy(p, yy)
            pz_gt = 1 - np.exp(-bec)
            noise_beta_b = (
                noise_beta_a
                * (1 - np.min([pz_gt, 1 - epsilon]))
                / np.max([pz_gt, epsilon])
            )
            pz = np.random.beta(noise_beta_a, noise_beta_b)
            absloss_gt[i] = pz_gt
            absloss_est[i] = pz
            grad = backward(xx, yy, pz)

        elif mode == "polyak_absloss-betanoise":
            p = forward(xx, theta)
            grad = backward(xx, yy, p)
            u = (2 * yy - 1) * np.dot(xx, theta)
            zeta = kappa * ratio(u) / (np.linalg.norm(xx) ** 2)
            zeta = min(zeta, kappa0)
            pz_gt = omega / (1 + np.exp(u))
            pz_gt = np.clip(pz_gt, a_min=epsilon, a_max=1 - epsilon)
            noise_beta_b = noise_beta_a * (1 - pz_gt) / pz_gt
            pz = np.random.beta(noise_beta_a, noise_beta_b)
            absloss_gt[i] = pz_gt
            absloss_est[i] = pz
            grad = zeta * grad / pz

        elif mode == "absloss-lr-refit":
            p = forward(xx, theta)
            pz_gt = abs(yy - p)
            xx_ext = np.concatenate((xx, [p]))
            try:
                pz = absloss_estimator.predict(xx_ext.reshape(1, -1))[0]
            except NotFittedError:
                pz = pz0
            absloss_gt[i] = pz_gt
            absloss_est[i] = pz
            pz = omega * pz
            pz = np.clip(pz, a_min=epsilon, a_max=1 - epsilon)
            grad = backward(xx, yy, p)

        elif mode == "polyak_absloss-lr-refit":
            p = forward(xx, theta)
            bec = cross_entropy(p, yy)
            grad = backward(xx, yy, p)
            zeta = kappa * bec / (np.linalg.norm(grad) ** 2)
            zeta = min(zeta, kappa0)
            pz_gt = 1 - np.exp(-bec)
            xx_ext = np.concatenate((xx, [p]))
            try:
                pz = absloss_estimator.predict(xx_ext.reshape(1, -1))[0]
            except NotFittedError:
                pz = pz0
            absloss_gt[i] = pz_gt
            absloss_est[i] = pz
            pz = np.clip(omega * pz, a_min=epsilon, a_max=1 - epsilon)
            grad = zeta * grad / pz

        # sampling
        pz = np.clip(pz, epsilon, 1 - epsilon)
        z = np.random.binomial(1, pz)

        losses[i] = loss(p, yy)
        losses_reg[i] = losses[i] + lam * np.linalg.norm(theta)

        accuracies[i] = accuracy(p, yy)

        labeled[i] = z
        probs[i] = pz
        # parameter update
        if z > 0:
            theta = update(theta, grad, eta=eta, lam=lam)
            if mode == "absloss-lr-refit" or mode == "polyak_absloss-lr-refit":
                absloss_xx_train.append(xx)
                absloss_yy_train.append(yy)
                if len(absloss_xx_train) > warmup_steps_estimator:
                    absloss_y_train = []
                    absloss_X_train = []
                    for _xx, _yy in zip(absloss_xx_train, absloss_yy_train):
                        _p = forward(_xx, theta)
                        absloss_y_train.append(abs(_yy - _p))
                        absloss_X_train.append(np.concatenate((_xx, [p])))
                    absloss_estimator.fit(absloss_X_train, absloss_y_train)

        p = forward(Xtest, theta)

        losses_test[i] = loss(p, ytest)
        losses_test_reg[i] = losses_test[i] + lam * np.linalg.norm(theta)
        accuracies_test[i] = accuracy(p, ytest)

        if verbose and i % 10 == 0:
            print(
                f"loss: {losses_test[i]}, train loss: {losses[i]}, ||theta||_1: {np.sum(np.abs(theta))} acc: {accuracies[i]}"
            )

    return {
        "losses": losses.tolist(),
        "losses_reg": losses_reg.tolist(),
        "losses_test": losses_test.tolist(),
        "losses_test_reg": losses_test_reg.tolist(),
        "labeled": labeled.tolist(),
        "theta": theta.tolist(),
        "probs": probs.tolist(),
        "accuracies": accuracies.tolist(),
        "accuracies_test": accuracies_test.tolist(),
    }
