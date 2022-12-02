from inspect import unwrap
import time
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.optim import SGD
from tqdm import tqdm  # type: ignore
from src.utils import ActivationsDataset


def rlace(
    ds: ActivationsDataset,
    dev_ds: Optional[ActivationsDataset] = None,
    n_dim: int = 1,
    device: str = "cpu",
    out_iters: int = 75000,
    in_iters_adv: int = 1,
    in_iters_clf: int = 1,
    epsilon: float = 0.0015,
    batch_size: int = 128,
    evalaute_every: int = 1000,
    optimizer_class=SGD,
    optimizer_params_P: Dict[str, Any] = {"lr": 0.005, "weight_decay": 1e-4},
    optimizer_params_predictor: Dict[str, Any] = {"lr": 0.005, "weight_decay": 1e-4},
    eval_clf_params: Dict[str, Any] = {
        "loss": "log_loss",
        "tol": 1e-4,
        "iters_no_change": 15,
        "alpha": 1e-4,
        "max_iter": 25000,
    },
    num_clfs_in_eval: int = 3,
) -> torch.Tensor:
    """Compute directions using RLACE.

    :param ds: An activation dataset containing the training data
    :param dev_ds: An activation dataset containing the validation data. If None, use training data.
    :param n_dim: Number of dimensions to neutralize from the input.
    :param device:
    :param out_iters: Number of batches to run
    :param in_iters_adv: number of iterations for adversary's optimization
    :param in_iters_clf: number of iterations from the predictor's optimization
    :param epsilon: stopping criterion. Stops if abs(acc - majority) < epsilon.
    :param batch_size:
    :param evalaute_every: After how many batches to evaluate the current adversary.
    :param optimizer_class: SGD/Adam etc.
    :param optimizer_params_P: P's optimizer's params (as a dict)
    :param optimizer_params_predictor: theta's optimizer's params (as a dict)
    :param eval_clf_params: the evaluation classifier params (as a dict)
    :param num_clfs_in_eval: the number of classifier trained for evaluation (change to 1 for large dataset / high dimensionality)

    RLACE by Ravfogel, 2022: see https://arxiv.org/pdf/2201.12091.pdf

    Adapted from https://github.com/shauli-ravfogel/rlace-icml/blob/2d9b6d03f65416172b4a2ca7f6da10e374002e5f/rlace.py
    """
    import sklearn  # type: ignore
    from sklearn.linear_model import SGDClassifier  # type: ignore

    hidden_dim = ds.x_data.shape[1]

    def init_classifier():
        return SGDClassifier(
            loss=eval_clf_params["loss"],
            fit_intercept=True,
            max_iter=eval_clf_params["max_iter"],
            tol=eval_clf_params["tol"],
            n_iter_no_change=eval_clf_params["iters_no_change"],
            n_jobs=32,
            alpha=eval_clf_params["alpha"],
        )

    def symmetric(X):
        X.data = 0.5 * (X.data + X.data.T)
        return X

    def get_score(X_np, y_np, X_dev, y_dev, P, n_dim):
        P_svd, _ = get_projection_and_dirs(P, n_dim)

        loss_vals = []
        accs = []

        for i in range(num_clfs_in_eval):
            clf = init_classifier()
            clf.fit(X_np @ P_svd, y_np)
            y_pred = clf.predict_proba(X_dev @ P_svd)
            loss = sklearn.metrics.log_loss(y_dev, y_pred)
            loss_vals.append(loss)
            accs.append(clf.score(X_dev @ P_svd, y_dev))

        i = np.argmin(loss_vals)
        return loss_vals[i], accs[i]

    def solve_constraint(lambdas, d=1):
        def f(theta):
            return_val = np.sum(np.minimum(np.maximum(lambdas - theta, 0), 1)) - d
            return return_val

        theta_min, theta_max = max(lambdas), min(lambdas) - 1
        assert f(theta_min) * f(theta_max) < 0
        mid = (theta_min + theta_max) / 2
        iters = 0
        while iters < 25:
            mid = (theta_min + theta_max) / 2
            if f(mid) * f(theta_min) > 0:
                theta_min = mid
            else:
                theta_max = mid
            iters += 1
        lambdas_plus = np.minimum(np.maximum(lambdas - mid, 0), 1)
        return lambdas_plus

    def get_majority_acc(y):
        c = Counter(y)
        fracts = [v / sum(c.values()) for v in c.values()]
        maj = max(fracts)
        return maj

    def get_loss_fn(X, y, predictor, P, bce_loss_fn, optimize_P=False):
        I = torch.eye(hidden_dim).to(device)
        bce = bce_loss_fn(predictor(X @ (I - P)).squeeze(), y)
        if optimize_P:
            bce = -bce
        return bce

    def get_projection_and_dirs(P, n_dim):
        _, U = np.linalg.eigh(P)
        U = U.T
        W = U[-n_dim:]
        return np.eye(P.shape[0]) - W.T @ W, W

    X_torch = ds.x_data.to(device)
    y_torch = ds.y_data.to(device)
    X_np = X_torch.cpu().numpy()
    y_np = y_torch.cpu().numpy()
    dev_ds_ = dev_ds or ds
    X_dev_np = dev_ds_.x_data.cpu().numpy()
    y_dev_np = dev_ds_.y_data.cpu().numpy()

    num_labels = len(set(y_np.tolist()))
    if num_labels == 2:
        predictor = torch.nn.Linear(hidden_dim, 1).to(device)
        bce_loss_fn: torch.nn.Module = torch.nn.BCEWithLogitsLoss()
        y_torch = y_torch.float()
    else:
        predictor = torch.nn.Linear(hidden_dim, num_labels).to(device)
        bce_loss_fn = torch.nn.CrossEntropyLoss()
        y_torch = y_torch.long()

    P = 1e-1 * torch.randn(hidden_dim, hidden_dim).to(device)
    P.requires_grad = True

    optimizer_predictor = optimizer_class(
        predictor.parameters(), **optimizer_params_predictor
    )
    optimizer_P = optimizer_class([P], **optimizer_params_P)

    maj = get_majority_acc(y_np)
    pbar = tqdm(range(out_iters), total=out_iters, ascii=True)
    count_examples = 0
    best_P, best_score, best_loss = None, 1, -1

    for i in pbar:

        for j in range(in_iters_adv):
            P = symmetric(P)
            optimizer_P.zero_grad()

            idx = np.arange(0, X_torch.shape[0])
            np.random.shuffle(idx)
            X_batch, y_batch = X_torch[idx[:batch_size]], y_torch[idx[:batch_size]]

            loss_P = get_loss_fn(
                X_batch, y_batch, predictor, symmetric(P), bce_loss_fn, optimize_P=True
            )
            loss_P.backward()
            optimizer_P.step()

            # project

            with torch.no_grad():
                D, U = torch.linalg.eigh(symmetric(P).detach().cpu())
                D = D.detach().cpu().numpy()
                D_plus_diag = solve_constraint(D, d=n_dim)
                D = torch.tensor(np.diag(D_plus_diag).real).float().to(device)
                U = U.to(device)
                P.data = U @ D @ U.T

        for j in range(in_iters_clf):
            optimizer_predictor.zero_grad()
            idx = np.arange(0, X_torch.shape[0])
            np.random.shuffle(idx)
            X_batch, y_batch = X_torch[idx[:batch_size]], y_torch[idx[:batch_size]]

            loss_predictor = get_loss_fn(
                X_batch, y_batch, predictor, symmetric(P), bce_loss_fn, optimize_P=False
            )
            loss_predictor.backward()
            optimizer_predictor.step()
            count_examples += batch_size

        if i % evalaute_every == 0:
            # pbar.set_description("Evaluating current adversary...")
            loss_val, score = get_score(
                X_np, y_np, X_dev_np, y_dev_np, P.detach().cpu().numpy(), n_dim
            )
            if (
                loss_val > best_loss
            ):  # if np.abs(score - maj) < np.abs(best_score - maj):
                best_P, best_loss = symmetric(P).detach().cpu().numpy().copy(), loss_val
            if np.abs(score - maj) < np.abs(best_score - maj):
                best_score = score

            # update progress bar

            best_so_far = (
                best_score if np.abs(best_score - maj) < np.abs(score - maj) else score
            )
            pbar.set_description(
                "{:.0f}/{:.0f}. Acc post-projection: {:.3f}%; best so-far: {:.3f}%; Maj: {:.3f}%; Gap: {:.3f}%; best loss: {:.4f}; current loss: {:.4f}".format(
                    i,
                    out_iters,
                    score * 100,
                    best_so_far * 100,
                    maj * 100,
                    np.abs(best_so_far - maj) * 100,
                    best_loss,
                    loss_val,
                )
            )
            pbar.refresh()  # to show immediately the update
            time.sleep(0.01)

        if i > 1 and np.abs(best_score - maj) < epsilon:
            break

    _, dirs = get_projection_and_dirs(best_P, n_dim)
    return torch.Tensor(dirs)


"""### Use RLACE to find the relevant dimensions"""
