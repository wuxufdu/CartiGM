"""PyTorch port of CellAssign's NB-mixture EM.

Reference R implementation lives at
``install/cellassign-master/R/inference-tensorflow.R``. CellAssign
upstream is unmaintained and its TF1 graph crashes under TF >= 2.
We re-implement the same EM in PyTorch so it can run on the same GPU
as the rest of the package.

Math mirrors ``inference_tensorflow``:

* per (n, g, c) NB mean
  ``mu_ngc = exp(X @ beta + log s + delta_gc * rho_gc)`` with the
  gradient through ``delta_gc`` stopped where ``rho_gc == 0``;
* per (n, g, c) dispersion ``phi_ngc`` is a sum of B Gaussian spline
  bumps over the data range;
* ``p_ngc = mu / (mu + phi)``; NB log-pmf matches
  ``tfp.distributions.NegativeBinomial(probs=p, total_count=phi)``;
* EM: E-step computes ``gamma_nc`` via softmax over c; M-step
  minimises ``-sum_n gamma_nc * log_p_n,c`` plus a Dirichlet prior
  on theta and (when shrinkage=True) a Normal prior on delta_log;
* ``delta_log`` is clamped to ``>= log(min_delta)`` after every Adam
  step exactly like the R ``constraint`` callback.

Memory: ``mu_ngc`` is float64 (N, G, C); for N=8000, G=171, C=10
this is ~110 MB; peak GPU memory stays well under 2 GB.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


__all__ = ["fit_cellassign_torch"]


def _nb_log_prob(y, total_count, probs):
    """NB log-pmf matching ``tfp.distributions.NegativeBinomial``."""
    eps = 1e-12
    y = y.to(probs.dtype)
    one_minus_p = torch.clamp(1.0 - probs, min=eps)
    p_clamped = torch.clamp(probs, min=eps)
    return (
        torch.lgamma(y + total_count)
        - torch.lgamma(y + 1.0)
        - torch.lgamma(total_count)
        + total_count * torch.log(one_minus_p)
        + y * torch.log(p_clamped)
    )


def _select_device(device):
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def fit_cellassign_torch(
    Y,
    rho,
    s,
    X=None,
    *,
    n_basis: int = 10,
    shrinkage: bool = True,
    learning_rate: float = 1e-2,
    max_iter_em: int = 20,
    max_iter_adam: int = 100,
    rel_tol_em: float = 1e-4,
    rel_tol_adam: float = 1e-4,
    min_delta: float = 2.0,
    dirichlet_concentration: float = 1e-2,
    random_seed: int = 0,
    device: Optional[str] = None,
    verbose: bool = False,
) -> dict:
    """Fit CellAssign's NB-mixture EM in PyTorch.

    Returns a dict with keys:
        ``cell_type`` int array of length N (argmax over C),
        ``gamma`` float64 (N, C) responsibilities,
        ``delta`` float64 (G, C) fitted log-fold change with rows
        zeroed where rho == 0,
        ``log_liks`` list[float] marginal log-likelihoods per EM
        iteration (first entry = initial ll before any M-step),
        ``n_iters`` int EM iterations completed,
        ``device`` str.
    """
    Y_np = np.asarray(Y, dtype=np.float64)
    rho_np = np.asarray(rho, dtype=np.float64)
    s_np = np.asarray(s, dtype=np.float64).reshape(-1)
    if Y_np.ndim != 2:
        raise ValueError("Y must be 2D (N, G); got shape " + str(Y_np.shape))
    if rho_np.ndim != 2:
        raise ValueError("rho must be 2D (G, C); got shape " + str(rho_np.shape))
    N, G = Y_np.shape
    G2, C = rho_np.shape
    if G != G2:
        raise ValueError("Y has " + str(G) + " genes but rho has " + str(G2))
    if s_np.shape[0] != N:
        raise ValueError("s must have length " + str(N) + "; got " + str(s_np.shape[0]))
    if X is None:
        X_np = np.ones((N, 1), dtype=np.float64)
    else:
        X_np = np.asarray(X, dtype=np.float64)
        if X_np.ndim == 1:
            X_np = X_np.reshape(-1, 1)
    if X_np.shape[0] != N:
        raise ValueError("X must have N rows; got shape " + str(X_np.shape))
    P = X_np.shape[1]

    device_t = _select_device(device)
    if verbose:
        print(
            "[cellassign-torch] device=" + str(device_t)
            + " N=" + str(N) + " G=" + str(G)
            + " C=" + str(C) + " P=" + str(P)
        )

    torch.manual_seed(int(random_seed))
    if device_t.type == "cuda":
        torch.cuda.manual_seed_all(int(random_seed))

    Y_t = torch.tensor(Y_np, dtype=torch.float64, device=device_t)
    rho_t = torch.tensor(rho_np, dtype=torch.float64, device=device_t)
    rho_bool = rho_t > 0.5
    s_t = torch.tensor(s_np, dtype=torch.float64, device=device_t)
    log_s_t = torch.log(torch.clamp(s_t, min=1e-12))
    X_t = torch.tensor(X_np, dtype=torch.float64, device=device_t)

    B = int(n_basis)
    y_min = float(Y_np.min())
    y_max = float(Y_np.max())
    if y_max <= y_min:
        y_max = y_min + 1.0
    basis_means_np = np.linspace(y_min, y_max, B)
    basis_means = torch.tensor(basis_means_np, dtype=torch.float64, device=device_t)
    b_init = 2.0 * (basis_means_np[1] - basis_means_np[0]) ** 2
    if b_init <= 0:
        b_init = 1.0
    b_const = torch.full((B,), 1.0 / b_init, dtype=torch.float64, device=device_t)

    LOWER_BOUND = 1e-10
    THETA_LOWER_BOUND = 1e-20
    log_min_delta = float(math.log(max(float(min_delta), 1e-12)))

    delta_log_init = torch.empty((G, C), dtype=torch.float64, device=device_t).uniform_(-2.0, 2.0)
    delta_log_init.clamp_(min=log_min_delta)
    delta_log = delta_log_init.clone().detach().requires_grad_(True)

    col_means = Y_np.mean(axis=0)
    cm_mean = float(col_means.mean())
    cm_std = float(col_means.std(ddof=1)) if col_means.size > 1 else 1.0
    if not np.isfinite(cm_std) or cm_std <= 0:
        cm_std = 1.0
    beta0 = (col_means - cm_mean) / cm_std
    beta_init = np.zeros((G, P), dtype=np.float64)
    beta_init[:, 0] = beta0
    beta = torch.tensor(beta_init, dtype=torch.float64, device=device_t, requires_grad=True)

    theta_logit = torch.empty(C, dtype=torch.float64, device=device_t).normal_().requires_grad_(True)
    a_log = torch.zeros(B, dtype=torch.float64, device=device_t, requires_grad=True)

    if shrinkage:
        delta_log_mean = torch.zeros((), dtype=torch.float64, device=device_t, requires_grad=True)
        delta_log_logvar = torch.zeros((), dtype=torch.float64, device=device_t, requires_grad=True)
        params = [delta_log, beta, theta_logit, a_log, delta_log_mean, delta_log_logvar]
    else:
        delta_log_mean = None
        delta_log_logvar = None
        params = [delta_log, beta, theta_logit, a_log]

    optimizer = torch.optim.Adam(params, lr=float(learning_rate))

    dirichlet_alpha = torch.full(
        (C,), float(dirichlet_concentration), dtype=torch.float64, device=device_t
    )

    def _forward():
        # Apply gradient stop where rho==0 to mimic ``entry_stop_gradients``
        # in the R reference: gradient flows through delta_log only at
        # marker positions (rho == 1).
        delta_log_eff = torch.where(rho_bool, delta_log, delta_log.detach())
        delta_log_eff = torch.clamp(delta_log_eff, min=log_min_delta)
        delta = torch.exp(delta_log_eff)

        base_mean = X_t @ beta.t() + log_s_t.unsqueeze(1)  # (N, G)
        mu_log = base_mean.unsqueeze(2) + (delta * rho_t).unsqueeze(0)  # (N, G, C)

        a = torch.exp(a_log)  # (B,)
        # phi_ngc = sum_b a_b * exp(-b_const_b * (mu_log - basis_means_b)^2) + LOWER_BOUND
        # diff: (N, G, C, B)
        diff = mu_log.unsqueeze(-1) - basis_means.view(1, 1, 1, B)
        kernel = torch.exp(-b_const.view(1, 1, 1, B) * diff * diff)
        phi = (a.view(1, 1, 1, B) * kernel).sum(dim=-1) + LOWER_BOUND  # (N, G, C)

        mu = torch.exp(mu_log)  # (N, G, C)
        p = mu / (mu + phi)

        log_pmf = _nb_log_prob(Y_t.unsqueeze(2), phi, p)  # (N, G, C)
        log_p_per_class = log_pmf.sum(dim=1)  # (N, C)

        theta_log = torch.log_softmax(theta_logit, dim=0)  # (C,)
        joint = log_p_per_class + theta_log.view(1, C)  # (N, C)

        # Marginal log likelihood per cell: logsumexp over c.
        marginal_ll = torch.logsumexp(joint, dim=1)  # (N,)
        return joint, theta_log, marginal_ll

    def _gamma_and_ll():
        with torch.no_grad():
            joint, theta_log, marginal_ll = _forward()
            log_gamma = joint - marginal_ll.unsqueeze(1)
            gamma = torch.exp(log_gamma)
            return gamma, marginal_ll.sum().item()

    def _theta_log_prob(theta_log):
        # Dirichlet log-prob for the simplex theta = exp(theta_log).
        theta = torch.exp(theta_log) + THETA_LOWER_BOUND
        log_norm = torch.lgamma(dirichlet_alpha.sum()) - torch.lgamma(dirichlet_alpha).sum()
        return log_norm + ((dirichlet_alpha - 1.0) * torch.log(theta)).sum()

    def _delta_log_prior_log_prob():
        if not shrinkage:
            return torch.zeros((), dtype=torch.float64, device=device_t)
        # Normal(loc = mean * rho, scale = sqrt(exp(logvar))) over (G, C).
        scale = torch.exp(0.5 * delta_log_logvar)
        scale = torch.clamp(scale, min=1e-6)
        loc = delta_log_mean * rho_t
        z = (delta_log - loc) / scale
        log_prob = -0.5 * z * z - torch.log(scale) - 0.5 * math.log(2.0 * math.pi)
        return log_prob.sum()

    log_liks = []
    gamma, ll0 = _gamma_and_ll()
    log_liks.append(ll0)
    if verbose:
        print("[cellassign-torch] EM init ll=" + str(ll0))

    n_iters = 0
    for em_step in range(int(max_iter_em)):
        # E-step: gamma already up to date from previous iteration.
        gamma_fixed = gamma.detach()

        Q_old = None
        for ai in range(int(max_iter_adam)):
            optimizer.zero_grad(set_to_none=True)
            joint, theta_log, _ = _forward()
            # M-step objective Q matches R: -sum gamma * joint + theta_prior + delta_prior.
            Q = -(gamma_fixed * joint).sum()
            theta_lp = _theta_log_prob(theta_log)
            Q = Q - theta_lp
            if shrinkage:
                delta_lp = _delta_log_prior_log_prob()
                Q = Q - delta_lp
            Q.backward()
            optimizer.step()
            with torch.no_grad():
                delta_log.clamp_(min=log_min_delta)

            if (ai + 1) % 20 == 0:
                Q_new = Q.detach().item()
                if Q_old is not None:
                    Q_diff = -(Q_new - Q_old) / max(abs(Q_old), 1e-12)
                    if Q_diff < float(rel_tol_adam):
                        if verbose:
                            print(
                                "[cellassign-torch] em=" + str(em_step + 1)
                                + " adam early stop at " + str(ai + 1)
                                + " Q_diff=" + str(Q_diff)
                            )
                        break
                Q_old = Q_new

        gamma, ll = _gamma_and_ll()
        log_liks.append(ll)
        n_iters += 1
        if verbose:
            print(
                "[cellassign-torch] em=" + str(em_step + 1)
                + " ll=" + str(ll)
            )
        ll_diff = (ll - log_liks[-2]) / max(abs(log_liks[-2]), 1e-12)
        if ll_diff < float(rel_tol_em):
            break

    with torch.no_grad():
        gamma, _ = _gamma_and_ll()
        delta_log_eff = torch.where(rho_bool, delta_log, delta_log.detach())
        delta_log_eff = torch.clamp(delta_log_eff, min=log_min_delta)
        delta = torch.exp(delta_log_eff)
        delta = torch.where(rho_bool, delta, torch.zeros_like(delta))

    cell_type = gamma.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
    return {
        "cell_type": cell_type,
        "gamma": gamma.detach().cpu().numpy(),
        "delta": delta.detach().cpu().numpy(),
        "log_liks": [float(x) for x in log_liks],
        "n_iters": int(n_iters),
        "device": str(device_t),
    }
