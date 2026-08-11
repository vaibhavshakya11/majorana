#!/usr/bin/env python3
"""Deterministic numerical audit of the RENP Dirac/Majorana tensor proposal.

MATHEMATICAL PURPOSE
====================
Radiative emission of a neutrino pair (RENP) leaves an invisible two-particle
system with total four-momentum ``q = p_i - p_f - k_gamma`` and invariant mass
``s = q^2``.  The proposed observable does not attempt to measure either
neutrino.  Instead, it reconstructs the positive-semidefinite polarization
response obtained after the neutrino momenta and spins have been integrated
out.  This file independently checks the algebraic tensor, its threshold rank,
the detector-level consequences, and several ways in which the ideal theorem
can fail or become difficult to observe.

CONVENTIONS AND EXACT TWO-BODY TENSOR
=====================================
All dimensionful inputs are in eV.  The metric is

    eta = diag(+1, -1, -1, -1),              s = q_mu q^mu,

and the leptonic current is

    L^mu = ubar_i gamma^mu P_L v_j,           P_L = (1-gamma5)/2.

For masses ``m1,m2``, define

    rho = sqrt([s-(m1+m2)^2][s-(m1-m2)^2]) / s,
    S_m = m1^2 + m2^2,                        D_m = m1^2 - m2^2,
    a = rho^2 - 3(1-S_m/s),
    b = 1 + S_m/s - 2 D_m^2/s^2.

With ``integral dPhi_2 = rho/(8*pi)`` and no identical-particle ``1/2``, the
spin-summed, phase-space-integrated Dirac LL tensor used here is

    T_D^{mu nu} = rho/(24*pi) [a s eta^{mu nu} + 2 b q^mu q^nu].

The chirality-interference trace needed for the Majorana exchange term is

    T_LR^{mu nu} = -rho m1 m2 eta^{mu nu}/(4*pi).

The widespread RENP convention ``gamma^mu(1-gamma5)`` multiplies these
``gamma^mu P_L`` spin-summed tensors by four.  Tests that concern normalized
eigenvalue slopes or ratios omit common positive prefactors explicitly.

For a diagonal equal-mass pair, ``m1=m2=m`` and
``beta=sqrt(1-4m^2/s)``, the matched tensors reduce to

    T_D = beta/(12*pi) [-(s-m^2) eta + (1+2m^2/s) q q],
    T_M = beta/(12*pi) [-(s-4m^2) eta + (1+2m^2/s) q q].

An experimental transfer matrix ``K`` maps the four-current into two analyzed
output-polarization amplitudes.  The observable response is therefore

    R_H = K T_H K^dagger,                     H in {D,M}.

The ``q q`` contribution has rank at most one.  For a coherent transfer with
two-dimensional support, the threshold powers ``delta=s-4m^2 -> 0+`` are

    Dirac:    lambda_min ~ delta^(1/2), lambda_max ~ delta^(1/2),
    Majorana: lambda_min ~ delta^(3/2), lambda_max ~ delta^(1/2).

Thus the leading Majorana response is rank one while the generic Dirac response
is rank two.  The basis-independent normalized determinant witness is

    W(R) = 4 det(R) / tr(R)^2.

Projecting onto the polarization vector orthogonal to ``K q`` eliminates the
rank-one recoil term.  For the diagonal channel its exact Majorana/Dirac ratio
is

    r_dark(beta) = 4 beta^2 / (3 + beta^2).

SCOPE OF THE NUMERICAL AUDIT
============================
The tests below deliberately attack the theorem from several directions:

1. construct gamma matrices and verify the Clifford algebra;
2. compare six-axis angular cubature and million-sample Monte Carlo integration
   of the spin trace with the covariant closed form in boosted frames;
3. fit all four diagonal threshold eigenvalue exponents and verify the exact
   recoil-dark ratio;
4. repeat the exponent test with a fully covariant complex ``2 x 4`` transfer;
5. show that an off-diagonal Majorana phase normally restores the ``1/2``
   exponent, isolating the special phase-aligned exception;
6. scan 500,000 masses, excess energies, and phases for positivity violations;
7. quantify how incoherent/multi-Kraus transfer can create an apparent rank-two
   Majorana response and verify the two-dimensional Cauchy--Binet identity;
8. evaluate pair-rest detuning, momentum spread, time-current leakage,
   resolution convolution, and leakage from the next neutrino threshold;
9. turn simple polarization probabilities into exact Chernoff information,
   finite-binomial decision errors, and seeded Stokes-witness ensembles; and
10. stress a stylized Poisson threshold-modulation fit against smooth
    backgrounds plus threshold-position and resolution nuisances.

The profile-likelihood calculation is explicitly a dimensionless design stress
test, not a target-specific RENP rate prediction and not a calibrated discovery
significance.  Its ``q`` values are Asimov shape separations per chosen
normalization only.

REPRODUCIBILITY WORKFLOW
========================
Every pseudorandom generator is local and explicitly seeded.  The seeds, in
test order, are ``20260810, 44117, 271828, 99, 8675309, 12345, 762341``.
Running

    python3 renp_numerical_stress.py

executes every assertion and writes ``renp_numerical_results.json`` beside this
file.  ``--skip-profiles`` omits only the comparatively slow nuisance-profile
study; ``--output PATH`` selects another JSON destination.  A nonzero exit or a
failed assertion means the audit did not reproduce.  JSON serialization is
sorted and indented, but floating-point last bits can still depend on the
NumPy/SciPy/BLAS versions; tolerances are chosen well above such roundoff.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize, minimize_scalar
from scipy.special import erfc, gamma as gamma_fn
from scipy.stats import binom, norm


ETA = np.diag([1.0, -1.0, -1.0, -1.0])


def py(value):
    """Recursively convert NumPy containers/scalars into JSON-native values."""
    if isinstance(value, dict):
        return {str(k): py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [py(v) for v in value]
    if isinstance(value, np.ndarray):
        return py(value.tolist())
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def gamma_matrices():
    """Return Dirac-basis gamma matrices and gamma5 for the ``+---`` metric."""
    i2 = np.eye(2, dtype=complex)
    z2 = np.zeros((2, 2), dtype=complex)
    sigmas = [
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex),
    ]
    gammas = [np.block([[i2, z2], [z2, -i2]])]
    gammas.extend(np.block([[z2, s], [-s, z2]]) for s in sigmas)
    gamma5 = 1j * gammas[0] @ gammas[1] @ gammas[2] @ gammas[3]
    return gammas, gamma5


GAMMA, GAMMA5 = gamma_matrices()
I4 = np.eye(4, dtype=complex)
PL = (I4 - GAMMA5) / 2
PR = (I4 + GAMMA5) / 2


def slash(p):
    """Form ``gamma^mu p_mu = gamma^0 p^0 - gamma^i p^i``."""
    return GAMMA[0] * p[0] - sum(GAMMA[a] * p[a] for a in (1, 2, 3))


def trace_current(p1, p2, m1, m2, mu, nu, left=True, right_left=True):
    """Evaluate an LL or LR spin trace before angular/phase-space integration.

    The antiparticle completeness relation supplies ``slash(p2)-m2``.  Selecting
    ``right_left=False`` replaces the final ``P_L`` by ``P_R`` and isolates the
    mass-dependent chirality interference entering Majorana exchange.
    """
    p_first = PL if left else PR
    p_second = PL if right_left else PR
    return np.trace(
        (slash(p1) + m1 * I4)
        @ GAMMA[mu]
        @ p_first
        @ (slash(p2) - m2 * I4)
        @ GAMMA[nu]
        @ p_second
    )


def boost(p, velocity):
    """Apply the active Lorentz boost used to test covariance away from rest."""
    velocity = np.asarray(velocity, dtype=float)
    b2 = velocity @ velocity
    if b2 == 0:
        return np.asarray(p, dtype=float).copy()
    factor = 1 / np.sqrt(1 - b2)
    bp = velocity @ p[1:]
    spatial = p[1:] + (((factor - 1) * bp / b2) + factor * p[0]) * velocity
    return np.r_[factor * (p[0] + bp), spatial]


def phase_rho(s, m1, m2):
    """Return the dimensionless Kallen factor ``sqrt(lambda)/s``."""
    return np.sqrt((s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2)) / s


def tensor_dirac_pl(q, m1, m2):
    """Return the exact integrated Dirac LL tensor for ``gamma^mu P_L``."""
    s = q @ ETA @ q
    sm = m1 * m1 + m2 * m2
    dm = m1 * m1 - m2 * m2
    rho = phase_rho(s, m1, m2)
    a = rho * rho - 3 * (1 - sm / s)
    b = 1 + sm / s - 2 * dm * dm / (s * s)
    return rho / (24 * np.pi) * (a * s * ETA + 2 * b * np.outer(q, q))


def tensor_lr_pl(s, m1, m2):
    """Return the exact integrated LR interference tensor."""
    return -phase_rho(s, m1, m2) * m1 * m2 * ETA / (4 * np.pi)


def six_axis_trace_tests():
    """Check Clifford algebra and exact covariant tensors by angular cubature.

    The six Cartesian directions integrate every angular polynomial of degree
    at most two exactly.  That is sufficient here because the spin trace is
    bilinear in the two on-shell momenta.  Three unequal/equal-mass boosted
    configurations prevent an accidental rest-frame-only agreement.
    """
    clifford = max(
        np.max(
            np.abs(
                GAMMA[i] @ GAMMA[j]
                + GAMMA[j] @ GAMMA[i]
                - 2 * ETA[i, j] * I4
            )
        )
        for i in range(4)
        for j in range(4)
    )
    axes = []
    for axis in range(3):
        for sign in (-1, 1):
            n = np.zeros(3)
            n[axis] = sign
            axes.append(n)
    cases = [
        (0.013, 0.051, 0.009, (0.31, -0.22, 0.17)),
        (0.011, 0.018, 0.001, (0.72, 0.10, -0.20)),
        (0.050, 0.050, 0.025, (0.00, 0.00, 0.83)),
    ]
    results = []
    for m1, m2, excess, velocity in cases:
        s = (m1 + m2) ** 2 + excess
        root_s = np.sqrt(s)
        e1 = (s + m1 * m1 - m2 * m2) / (2 * root_s)
        e2 = root_s - e1
        momentum = np.sqrt(
            (s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2)
        ) / (2 * root_s)
        velocity = np.asarray(velocity)
        factor = 1 / np.sqrt(1 - velocity @ velocity)
        q = np.r_[factor * root_s, factor * root_s * velocity]
        ll = np.zeros((4, 4), dtype=complex)
        lr = np.zeros((4, 4), dtype=complex)
        for n in axes:
            p1 = boost(np.r_[e1, momentum * n], velocity)
            p2 = boost(np.r_[e2, -momentum * n], velocity)
            for mu in range(4):
                for nu in range(4):
                    ll[mu, nu] += trace_current(
                        p1, p2, m1, m2, mu, nu, left=True, right_left=True
                    )
                    lr[mu, nu] += trace_current(
                        p1, p2, m1, m2, mu, nu, left=True, right_left=False
                    )
        # The equally weighted six directions already represent the solid-angle
        # average; multiplying by total Phi_2 completes the phase-space integral.
        phi2 = phase_rho(s, m1, m2) / (8 * np.pi)
        ll *= phi2 / len(axes)
        lr *= phi2 / len(axes)
        exact_ll = tensor_dirac_pl(q, m1, m2)
        exact_lr = tensor_lr_pl(s, m1, m2)
        scale_ll = np.max(np.abs(exact_ll))
        scale_lr = np.max(np.abs(exact_lr))
        results.append(
            {
                "masses": [m1, m2],
                "s": s,
                "boost": velocity,
                "q": q,
                "relative_LL_error": np.max(np.abs(ll.real - exact_ll)) / scale_ll,
                "relative_LR_error": np.max(np.abs(lr.real - exact_lr)) / scale_lr,
                "residual_imaginary_over_scale": np.max(np.abs(ll.imag)) / scale_ll,
            }
        )
    assert clifford < 1e-14
    assert max(r["relative_LL_error"] for r in results) < 2e-12
    assert max(r["relative_LR_error"] for r in results) < 2e-12
    assert max(r["residual_imaginary_over_scale"] for r in results) < 2e-12
    return {"clifford_max_error": clifford, "cases": results}


def monte_carlo_phase_space():
    """Independently recover the covariant tensor by uniform sphere sampling.

    The test uses the ``gamma(1-gamma5)`` trace identity, hence the explicit
    factor four relative to the ``P_L`` analytic tensor.  Reporting
    ``sqrt(N)*error`` checks the expected Monte Carlo convergence law as well as
    the final absolute accuracy.
    """
    rng = np.random.default_rng(20260810)
    m1, m2 = 0.013, 0.051
    s = (m1 + m2) ** 2 + 0.009
    root_s = np.sqrt(s)
    velocity = np.array([0.31, -0.22, 0.17])
    factor = 1 / np.sqrt(1 - velocity @ velocity)
    q = np.r_[factor * root_s, factor * root_s * velocity]
    e1 = (s + m1 * m1 - m2 * m2) / (2 * root_s)
    e2 = root_s - e1
    momentum = np.sqrt(
        (s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2)
    ) / (2 * root_s)
    exact = 4 * tensor_dirac_pl(q, m1, m2)  # gamma(1-gamma5) convention
    rho = phase_rho(s, m1, m2)

    def boost_many(momentum_array):
        """Vectorized counterpart of :func:`boost` for the million-point run."""
        bp = momentum_array[:, 1:] @ velocity
        out = np.empty_like(momentum_array)
        out[:, 0] = factor * (momentum_array[:, 0] + bp)
        out[:, 1:] = momentum_array[:, 1:] + (
            ((factor - 1) * bp / (velocity @ velocity) + factor * momentum_array[:, 0])[
                :, None
            ]
            * velocity
        )
        return out

    output = []
    for count in (1_000, 10_000, 100_000, 1_000_000):
        z = rng.uniform(-1, 1, count)
        phi = rng.uniform(0, 2 * np.pi, count)
        radial = np.sqrt(1 - z * z)
        direction = np.c_[radial * np.cos(phi), radial * np.sin(phi), z]
        p1 = boost_many(np.c_[np.full(count, e1), momentum * direction])
        p2 = boost_many(np.c_[np.full(count, e2), -momentum * direction])
        dot = np.einsum("ni,ij,nj->n", p1, ETA, p2)
        average = np.empty((4, 4))
        for mu in range(4):
            for nu in range(4):
                average[mu, nu] = 8 * np.mean(
                    p1[:, mu] * p2[:, nu]
                    + p1[:, nu] * p2[:, mu]
                    - ETA[mu, nu] * dot
                )
        estimate = rho / (8 * np.pi) * average
        error = np.linalg.norm(estimate - exact) / np.linalg.norm(exact)
        output.append(
            {"samples": count, "relative_frobenius_error": error, "sqrtN_error": np.sqrt(count) * error}
        )
    assert output[-1]["relative_frobenius_error"] < 1e-3
    assert 0.1 < np.mean([row["sqrtN_error"] for row in output]) < 1.0
    return output


def log_slope(x, y, mask):
    """Fit a power-law exponent from ``log(y)`` versus ``log(x)``."""
    return np.polyfit(np.log(x[mask]), np.log(np.asarray(y)[mask]), 1)[0]


def threshold_tests():
    """Verify the diagonal D/M rank theorem and exact recoil-dark ratio.

    Here ``x=(s-4m^2)/m^2``.  ``gram=K K^dagger`` carries the isotropic term and
    ``recoil=K q`` carries the rank-one term.  The random complex transfer is
    fixed and full rank; its precise values must not affect universal powers.
    """
    rng = np.random.default_rng(44117)
    m = 0.010
    transfer = rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3))
    transfer /= np.linalg.norm(transfer, axis=1)[:, None]
    q_spatial = np.array([0.021, -0.013, 0.017])
    gram = transfer @ transfer.conj().T
    recoil = transfer @ q_spatial
    x = np.logspace(-10, -3, 180)
    mask = x < 1e-5
    eigenvalues = {key: [] for key in ("D_min", "D_max", "M_min", "M_max")}
    witnesses = {"D": [], "M": []}
    for value in x:
        s = 4 * m * m + m * m * value
        beta = np.sqrt(1 - 4 * m * m / s)
        coefficient = 1 + 2 * m * m / s
        for hypothesis, isotropic in (("D", s - m * m), ("M", s - 4 * m * m)):
            # A common positive normalization is irrelevant to eigenvalue powers,
            # determinant witness, and the dark-channel D/M ratio.
            response = beta * (
                isotropic * gram + coefficient * np.outer(recoil, recoil.conj())
            )
            eig = np.linalg.eigvalsh(response)
            eigenvalues[f"{hypothesis}_min"].append(eig[0])
            eigenvalues[f"{hypothesis}_max"].append(eig[1])
            witnesses[hypothesis].append(
                4 * np.linalg.det(response).real / np.trace(response).real ** 2
            )
    slopes = {key: log_slope(x, values, mask) for key, values in eigenvalues.items()}
    assert abs(slopes["D_min"] - 0.5) < 2e-5
    assert abs(slopes["D_max"] - 0.5) < 2e-5
    assert abs(slopes["M_min"] - 1.5) < 2e-5
    assert abs(slopes["M_max"] - 0.5) < 2e-5

    # This complex two-vector is exactly Hermitian-orthogonal to K q.
    dark = np.array([-np.conj(recoil[1]), np.conj(recoil[0])])
    dark /= np.linalg.norm(dark)
    dark_rows = []
    for beta in (0.01, 0.05, 0.15, 0.30, 0.70):
        s = 4 * m * m / (1 - beta * beta)
        coefficient = 1 + 2 * m * m / s
        rd = beta * ((s - m * m) * gram + coefficient * np.outer(recoil, recoil.conj()))
        rm = beta * ((s - 4 * m * m) * gram + coefficient * np.outer(recoil, recoil.conj()))
        measured = (np.vdot(dark, rm @ dark) / np.vdot(dark, rd @ dark)).real
        exact = 4 * beta * beta / (3 + beta * beta)
        dark_rows.append({"beta": beta, "measured": measured, "exact": exact, "absolute_error": abs(measured - exact)})
    assert max(row["absolute_error"] for row in dark_rows) < 2e-13
    return {
        "slopes": slopes,
        "gram_eigenvalues": np.linalg.eigvalsh(gram),
        "recoil_norm_squared": np.vdot(recoil, recoil).real,
        "witness_smallest_x": {key: values[0] for key, values in witnesses.items()},
        "dark_orthogonality": abs(np.vdot(dark, recoil)),
        "dark_ratio": dark_rows,
    }


def covariant_threshold_test():
    """Repeat the eigenvalue-power test with temporal currents and a large boost."""
    rng = np.random.default_rng(271828)
    m = 0.010
    velocity = np.array([0.82, -0.18, 0.22])
    factor = 1 / np.sqrt(1 - velocity @ velocity)
    transfer = rng.normal(size=(2, 4)) + 1j * rng.normal(size=(2, 4))
    transfer /= np.linalg.norm(transfer, axis=1)[:, None]
    x = np.logspace(-10, -3, 180)
    mask = x < 1e-5
    values = {key: [] for key in ("D_min", "D_max", "M_min", "M_max")}
    for value in x:
        s = m * m * (4 + value)
        root_s = np.sqrt(s)
        q = np.r_[factor * root_s, factor * root_s * velocity]
        beta = np.sqrt(1 - 4 * m * m / s)
        td = beta / (12 * np.pi) * (
            -(s - m * m) * ETA + (1 + 2 * m * m / s) * np.outer(q, q)
        )
        tm = beta / (12 * np.pi) * (
            -(s - 4 * m * m) * ETA + (1 + 2 * m * m / s) * np.outer(q, q)
        )
        for hypothesis, tensor in (("D", td), ("M", tm)):
            eig = np.linalg.eigvalsh(transfer @ tensor @ transfer.conj().T)
            values[f"{hypothesis}_min"].append(eig[0])
            values[f"{hypothesis}_max"].append(eig[1])
    slopes = {key: log_slope(x, val, mask) for key, val in values.items()}
    assert abs(slopes["D_min"] - 0.5) < 2e-5
    assert abs(slopes["M_min"] - 1.5) < 2e-5
    return {"boost_speed": np.linalg.norm(velocity), "slopes": slopes}


def off_diagonal_tests():
    """Test unequal-mass thresholds and dependence on the Majorana phase.

    ``K_M=K_D-6 m1 m2 cos(2 phi)`` is the isotropic coefficient after Majorana
    exchange.  Only the phase-aligned ``phi=0`` case cancels at threshold and
    retains the ``delta^(3/2)`` small eigenvalue; generic phases do not.
    """
    m1, m2 = 0.010, 0.014
    s0 = (m1 + m2) ** 2
    q = np.array([0.017, -0.011, 0.013])
    rng = np.random.default_rng(99)
    transfer = rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3))
    gram = transfer @ transfer.conj().T
    recoil = transfer @ q
    x = np.logspace(-11, -4, 160)
    mask = x < 1e-6
    output = []
    for phase in (0, np.pi / 12, np.pi / 4, np.pi / 2):
        minimum = []
        ratios = []
        for value in x:
            s = s0 + m1 * m2 * value
            rho = phase_rho(s, m1, m2)
            sm = m1 * m1 + m2 * m2
            dm = m1 * m1 - m2 * m2
            kd = 2 * s - sm - dm * dm / s
            b = 1 + sm / s - 2 * dm * dm / (s * s)
            km = kd - 6 * np.cos(2 * phase) * m1 * m2
            response = rho * (km * gram + 2 * b * np.outer(recoil, recoil.conj()))
            minimum.append(np.linalg.eigvalsh(response)[0])
            ratios.append(km / kd)
        row = {
            "phase_over_pi": phase / np.pi,
            "minimum_eigenvalue_exponent": log_slope(x, minimum, mask),
            "threshold_isotropic_ratio_analytic": 2 * np.sin(phase) ** 2,
            "threshold_isotropic_ratio_numeric": ratios[0],
        }
        output.append(row)
    assert abs(output[0]["minimum_eigenvalue_exponent"] - 1.5) < 2e-5
    assert all(abs(row["minimum_eigenvalue_exponent"] - 0.5) < 2e-5 for row in output[1:])
    assert max(abs(row["threshold_isotropic_ratio_analytic"] - row["threshold_isotropic_ratio_numeric"]) for row in output) < 2e-9
    return output


def positivity_scan():
    """Search a broad random domain for forbidden negative response coefficients."""
    rng = np.random.default_rng(8675309)
    count = 500_000
    m1 = 10 ** rng.uniform(-4, -0.7, count)
    m2 = 10 ** rng.uniform(-4, -0.7, count)
    x = 10 ** rng.uniform(-12, 3, count)
    s = (m1 + m2) ** 2 + m1 * m2 * x
    phase = rng.uniform(0, 2 * np.pi, count)
    sm = m1 * m1 + m2 * m2
    dm = m1 * m1 - m2 * m2
    kd = 2 * s - sm - dm * dm / s
    km = kd - 6 * np.cos(2 * phase) * m1 * m2
    b = 1 + sm / s - 2 * dm * dm / (s * s)
    result = {
        "samples": count,
        "min_KD_over_mimj": np.min(kd / (m1 * m2)),
        "min_KM_over_mimj": np.min(km / (m1 * m2)),
        "min_q_coefficient": np.min(b),
        "fraction_KM_greater_KD": np.mean(km > kd),
        "KM_over_KD_range": [np.min(km / kd), np.max(km / kd)],
    }
    assert result["min_KD_over_mimj"] >= 6 - 1e-9
    assert result["min_KM_over_mimj"] >= -1e-9
    assert result["min_q_coefficient"] >= -1e-12
    return result


def kraus_tests():
    """Quantify rank inflation from incoherent transfer paths.

    A sum of individually rank-one output matrices need not remain rank one.
    The final equality checks the 2D Cauchy--Binet determinant formula directly,
    making this a counterexample audit rather than an idealized robustness claim.
    """
    table = []
    for weight in (1e-4, 1e-3, 0.01, 0.1, 1.0):
        witnesses = []
        for degrees in (5, 15, 30, 60, 90):
            value = 4 * weight * np.sin(np.deg2rad(degrees)) ** 2 / (1 + weight) ** 2
            witnesses.append({"angle_degrees": degrees, "witness": value})
        table.append({"relative_weight": weight, "values": witnesses})
    rng = np.random.default_rng(12345)
    vectors = rng.normal(size=(5, 2)) + 1j * rng.normal(size=(5, 2))
    weights = rng.uniform(0.1, 2, 5)
    matrix = sum(w * np.outer(c, c.conj()) for w, c in zip(weights, vectors))
    cauchy_binet = sum(
        weights[r] * weights[t] * abs(np.linalg.det(np.stack([vectors[r], vectors[t]], axis=1))) ** 2
        for r in range(5)
        for t in range(r + 1, 5)
    )
    error = abs(np.linalg.det(matrix).real - cauchy_binet)
    assert error < 2e-12
    return {"two_path_witness_table": table, "cauchy_binet_absolute_error": error}


def pair_rest_tests():
    """Tabulate pair-rest tuning tolerances and leading transfer-contamination terms."""
    mass = 0.010
    e_eg = 1.0
    detunings = []
    for beta in (0.02, 0.05, 0.10, 0.15, 0.30):
        s = 4 * mass * mass / (1 - beta * beta)
        epsilon = np.sqrt(s) - 2 * mass
        detunings.append(
            {"beta": beta, "delta_s_eV2": s - 4 * mass * mass, "q0_detuning_meV": epsilon * 1e3}
        )
    fixed_beta = 0.15
    s = 4 * mass * mass / (1 - fixed_beta * fixed_beta)
    momentum_mistuning = []
    for q_mev in (0, 0.5, 1, 2, 3, 5):
        q_abs = q_mev * 1e-3
        epsilon = np.sqrt(s + q_abs * q_abs) - 2 * mass
        momentum_mistuning.append({"q_mismatch_meV": q_mev, "q0_detuning_meV": epsilon * 1e3})
    spread = []
    for beta in (0.02, 0.05, 0.10, 0.15, 0.30):
        s = 4 * mass * mass / (1 - beta * beta)
        v = 1 + 2 * mass * mass / s
        values = []
        for sigma_over_m in (0, 0.05, 0.10, 0.25, 0.50):
            # An isotropic 3D momentum variance contributes one third per axis.
            variance_per_axis = (sigma_over_m * mass) ** 2 / 3
            ratio = ((s - 4 * mass * mass) + v * variance_per_axis) / (
                (s - mass * mass) + v * variance_per_axis
            )
            values.append({"sigma_Q_over_m": sigma_over_m, "ratio": ratio})
        spread.append({"beta": beta, "values": values})
    bounds = []
    for leakage in (0.001, 0.01, 0.03, 0.10):
        sigma_over_m = np.sqrt(6 * leakage / (1 - leakage))
        bounds.append(
            {
                "leakage": leakage,
                "max_sigma_Q_over_m": sigma_over_m,
                "max_sigma_Q_meV_for_10meV_mass": 10 * sigma_over_m,
            }
        )
    time_current = []
    for epsilon in (0.01, 0.03, 0.10, 0.30):
        time_current.append(
            {"time_to_spatial_norm": epsilon, "threshold_ratio": 2 * epsilon * epsilon / (1 + epsilon * epsilon)}
        )
    return {
        "omega0_and_peg_eV": e_eg - 2 * mass,
        "energy_detunings": detunings,
        "momentum_mistuning_at_beta_0p15": momentum_mistuning,
        "isotropic_spread": spread,
        "beta_to_zero_spread_bounds": bounds,
        "pair_rest_time_current_leakage": time_current,
    }


def smearing_tests():
    """Convolve ideal threshold powers with Gaussian instrumental resolution.

    The local logarithmic slope shows how far above a blurred edge one must scan
    before the asymptotic ``1/2`` or ``3/2`` law re-emerges.  The second table
    measures contamination leaking in from a neighboring, higher threshold.
    """
    sigma = 0.01
    dx = 2e-5
    x = np.arange(-0.2, 0.5 + dx, dx)
    ratios = (0.05, 0.10, 0.25, 0.50, 1, 2, 5, 10)
    output = {}
    for alpha in (0.5, 1.5):
        raw = np.maximum(x, 0) ** alpha
        smeared = gaussian_filter1d(raw, sigma / dx, mode="constant", truncate=8)
        derivative = np.gradient(smeared, dx)
        rows = []
        for ratio in ratios:
            y = ratio * sigma
            index = np.argmin(abs(x - y))
            rows.append(
                {
                    "x_over_sigma": ratio,
                    "effective_log_slope": y * derivative[index] / smeared[index],
                    "smeared_over_raw": smeared[index] / y ** alpha,
                }
            )
        constant_numeric = quad(lambda z: z ** alpha * norm.pdf(z), 0, np.inf, epsabs=1e-13)[0]
        constant_exact = 2 ** ((alpha - 2) / 2) * gamma_fn((alpha + 1) / 2) / np.sqrt(np.pi)
        # scipy.stats' tail quadrature is accurate to about 1e-12 for alpha=3/2.
        assert abs(constant_numeric - constant_exact) < 2e-11
        output[str(alpha)] = {
            "edge_constant": constant_exact,
            "local_slopes": rows,
        }
    leakage = {}
    for alpha in (0.5, 1.5):
        denominator = quad(lambda z: z ** alpha * norm.pdf(z), 0, np.inf, epsabs=1e-13)[0]
        rows = []
        for distance in (1, 2, 2.326347874, 3, 3.090232306, 4, 4.753424309, 5):
            numerator = quad(
                lambda z: (z - distance) ** alpha * norm.pdf(z),
                distance,
                np.inf,
                epsabs=1e-13,
            )[0]
            rows.append(
                {
                    "threshold_gap_over_sigma": distance,
                    "gaussian_tail_probability": 0.5 * erfc(distance / np.sqrt(2)),
                    "relative_convolved_onset": numerator / denominator,
                }
            )
        leakage[str(alpha)] = rows
    return {"sigma_used_for_local_slope": sigma, "power_law_convolution": output, "next_threshold_leakage": leakage}


def threshold_gap_tests():
    """Convert oscillation mass splittings into adjacent-threshold resolution scales."""
    solar = 7.49e-5
    atmospheric_normal = 2.513e-3
    atmospheric_inverted = 2.484e-3
    rows = []
    for ordering in ("NO", "IO"):
        for lightest_mev in (0, 1, 10, 50, 100):
            lightest = lightest_mev / 1000
            if ordering == "NO":
                masses = np.array(
                    [lightest, np.sqrt(lightest * lightest + solar), np.sqrt(lightest * lightest + atmospheric_normal)]
                )
            else:
                masses = np.array(
                    [
                        np.sqrt(lightest * lightest + atmospheric_inverted),
                        np.sqrt(lightest * lightest + atmospheric_inverted + solar),
                        lightest,
                    ]
                )
            first = int(np.argmin(masses))
            candidates = [index for index in range(3) if index != first]
            second = min(candidates, key=lambda index: (masses[first] + masses[index]) ** 2)
            delta_s = (masses[first] + masses[second]) ** 2 - (2 * masses[first]) ** 2
            # Linearized unboosted mapping for Eeg=1 eV: ds/domega=-2 Eeg.
            photon_gap_micro_eV = delta_s / 2 * 1e6
            rows.append(
                {
                    "ordering": ordering,
                    "lightest_meV": lightest_mev,
                    "first_index": first + 1,
                    "next_index": second + 1,
                    "delta_s_eV2": delta_s,
                    "photon_gap_micro_eV_for_Eeg_1eV": photon_gap_micro_eV,
                    "sigma_micro_eV_for_1pct_tail": photon_gap_micro_eV / 2.326347874,
                    "sigma_micro_eV_for_1ppm_tail": photon_gap_micro_eV / 4.753424309,
                }
            )
    return rows


def chernoff_bernoulli(p, q):
    """Return Bernoulli Chernoff information and its optimizing interpolation."""
    pvec = np.array([p, 1 - p])
    qvec = np.array([q, 1 - q])
    result = minimize_scalar(
        lambda t: np.sum(pvec ** t * qvec ** (1 - t)),
        bounds=(0, 1),
        method="bounded",
        options={"xatol": 1e-14},
    )
    return -np.log(result.fun), result.x


def bayes_error_binomial(count, p, q):
    """Compute the exact equal-prior Bayes error for two binomial laws."""
    outcomes = np.arange(count + 1)
    return 0.5 * np.sum(
        np.minimum(binom.pmf(outcomes, count, p), binom.pmf(outcomes, count, q))
    )


def events_needed(p, q, target):
    """Find the smallest informative-event count attaining a target Bayes error."""
    high = 1
    while bayes_error_binomial(high, p, q) > target:
        high *= 2
    low = 1
    while low < high:
        middle = (low + high) // 2
        if bayes_error_binomial(middle, p, q) <= target:
            high = middle
        else:
            low = middle + 1
    return low


def tomography_tests():
    """Propagate leakage/depolarization through exact tests and Stokes tomography.

    The first calculation is an exact binary-outcome benchmark.  The seeded
    ensemble then estimates a bias-corrected determinant witness from three
    Pauli measurement bases and selects an equal-prior classification threshold.
    """
    scenarios = [
        (0.00, 0.00),
        (0.01, 0.00),
        (0.03, 0.00),
        (0.01, 0.05),
        (0.03, 0.20),
        (0.10, 0.20),
    ]
    targets = (0.05, 0.00135, 2.87e-7)
    rows = []
    for leakage, depolarization in scenarios:
        p_dirac = (1 - depolarization) * 0.8 + depolarization * 0.5
        p_majorana = (1 - depolarization) * (1 - leakage) + depolarization * 0.5
        information, t_star = chernoff_bernoulli(p_dirac, p_majorana)
        informative = [events_needed(p_dirac, p_majorana, target) for target in targets]
        rows.append(
            {
                "majorana_leakage": leakage,
                "depolarization": depolarization,
                "p_dirac": p_dirac,
                "p_majorana": p_majorana,
                "chernoff_per_informative_event": information,
                "chernoff_t": t_star,
                "informative_events_95_3sigma_5sigma": informative,
                "total_six_projector_events_95_3sigma_5sigma": [3 * n for n in informative],
            }
        )
    rng = np.random.default_rng(762341)
    simulations = []
    replicates = 200_000
    for total in (90, 300):
        per_basis = total // 3
        witnesses = {}
        for label, probability in (("D", 0.8), ("M", 0.99)):
            counts = rng.binomial(
                per_basis,
                np.array([probability, 0.5, 0.5]),
                size=(replicates, 3),
            )
            stokes = 2 * counts / per_basis - 1
            # Subtract the known finite-binomial variance before forming |S|^2;
            # for a normalized 2x2 response, W=1-|S|^2.
            witness = 1 - np.sum(
                (stokes * stokes - 1 / per_basis) / (1 - 1 / per_basis), axis=1
            )
            witnesses[label] = witness
        candidates = np.unique(
            np.quantile(np.r_[witnesses["D"], witnesses["M"]], np.linspace(0, 1, 2001))
        )
        errors = np.array(
            [
                0.5
                * (
                    np.mean(witnesses["D"] <= threshold)
                    + np.mean(witnesses["M"] > threshold)
                )
                for threshold in candidates
            ]
        )
        best = int(np.argmin(errors))
        simulations.append(
            {
                "total_events": total,
                "replicates": replicates,
                "D_median": np.median(witnesses["D"]),
                "D_5_95": np.quantile(witnesses["D"], [0.05, 0.95]),
                "M_median": np.median(witnesses["M"]),
                "M_5_95": np.quantile(witnesses["M"], [0.05, 0.95]),
                "best_threshold": candidates[best],
                "equal_prior_error": errors[best],
            }
        )
    assert rows[1]["total_six_projector_events_95_3sigma_5sigma"] == [57, 192, 525]
    return {"exact_binomial": rows, "seeded_witness_simulation": simulations}


def high_boost_tests():
    """Expose the tradeoff between recoil brightness and dark-channel occupancy."""
    rows = []
    five_sigma_tail = 2.87e-7
    for q_over_m in (0, 1, 3, 10, 30, 100):
        ratio = 0.5 * q_over_m * q_over_m
        witness = 4 * (1 + ratio) / (2 + ratio) ** 2
        dark_fraction = 1 / (2 + ratio)
        count = np.log(0.5 / five_sigma_tail) / (-np.log(1 - dark_fraction))
        rows.append(
            {
                "Q_over_m": q_over_m,
                "bright_to_isotropic_ratio": ratio,
                "Dirac_rank_witness": witness,
                "dark_fraction": dark_fraction,
                "ideal_passive_events_for_5sigma": count,
            }
        )
    return rows


def pmns_diagonal_weights():
    """Evaluate diagonal electron-current weights at the stated PMNS benchmark."""
    # NuFIT 6.1 normal-ordering best fit without the tabulated SK-atmospheric
    # likelihood (data available in November 2025), matching Appendix H.
    sin2_12 = 0.3088
    sin2_13 = 0.02249
    components = [
        (1 - sin2_12) * (1 - sin2_13),
        sin2_12 * (1 - sin2_13),
        sin2_13,
    ]
    return [
        {"index": index + 1, "Uei_squared": value, "aii": value - 0.5, "aii_squared": (value - 0.5) ** 2}
        for index, value in enumerate(components)
    ]


def profile_modulation_test():
    """Profile a stylized threshold scan against backgrounds and response nuisances.

    Counts are expressed per arbitrary Dirac-normalized exposure.  The returned
    Asimov Poisson deviances establish numerical non-degeneracy inside this toy
    nuisance family; they are neither calibrated significances nor event rates.
    """
    settings = np.r_[
        -0.6,
        -0.4,
        -0.25,
        -0.12,
        -0.04,
        np.geomspace(0.015, 3, 22),
    ]
    standardized = (settings - settings.mean()) / (settings.max() - settings.min())
    dx = 0.001
    grid = np.arange(-1.5, 5, dx)

    def raw(hypothesis):
        """Ideal dimensionless diagonal dark-channel onset before smearing."""
        positive = np.maximum(grid, 0)
        beta = np.sqrt(positive / (4 + positive))
        return beta * ((3 + positive) if hypothesis == "D" else positive)

    raw_signal = {hypothesis: raw(hypothesis) for hypothesis in ("D", "M")}

    def unnormalized(hypothesis, shift=0, sigma=0.025):
        """Apply Gaussian resolution and interpolate at experimental settings."""
        convolved = gaussian_filter1d(
            raw_signal[hypothesis], sigma / dx, mode="constant", truncate=7
        )
        return np.interp(settings - shift, grid, convolved)

    normalization = unnormalized("D").sum()

    def signal(hypothesis, shift=0, sigma=0.025):
        """Normalize both hypotheses to the same reference Dirac exposure."""
        return unnormalized(hypothesis, shift, sigma) / normalization

    def profile(true_hypothesis, alternative, background_fraction, degree, response_nuisance):
        """Minimize the Poisson Asimov deviance of one hypothesis against the other."""
        data = signal(true_hypothesis) + background_fraction / len(settings)
        parameters = [0.0, np.log(background_fraction)] + [0.0] * degree
        bounds = [(-5, 5), (-12, np.log(100))] + [(-2, 2)] * degree
        if response_nuisance:
            parameters.extend([0.0, np.log(0.025)])
            bounds.extend([(-0.025, 0.025), (np.log(0.015), np.log(0.04))])

        def model(values):
            """Build the profiled alternative mean for one nuisance vector."""
            # Exponentiating a low-order polynomial keeps the profiled smooth
            # background nonnegative without imposing a fixed overall scale.
            polynomial = np.zeros_like(standardized)
            for power in range(1, degree + 1):
                polynomial += values[1 + power] * standardized ** power
            shape = np.exp(polynomial - polynomial.max())
            background = np.exp(values[1]) * shape / shape.sum()
            index = 2 + degree
            if response_nuisance:
                shift, sigma = values[index], np.exp(values[index + 1])
            else:
                shift, sigma = 0.0, 0.025
            return np.exp(values[0]) * signal(alternative, shift, sigma) + background

        def deviance(values):
            """Return twice the Poisson KL divergence from truth to the model."""
            candidate = np.maximum(model(values), 1e-15)
            return 2 * np.sum(candidate - data + data * np.log(data / candidate))

        best = None
        # Multiple signal-normalization starts reduce vulnerability to a local
        # optimum in the non-linear nuisance surface.
        for scale_start in (-0.5, 0, 0.5):
            start = np.array(parameters)
            start[0] += scale_start
            fit = minimize(
                deviance,
                start,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 800, "ftol": 1e-12},
            )
            if best is None or fit.fun < best.fun:
                best = fit
        assert best.success or best.fun < 1e-3
        return best.fun

    rows = []
    for true_hypothesis, alternative in (("D", "M"), ("M", "D")):
        for background_fraction in (0.1, 1.0, 10.0):
            for degree, response_nuisance in ((0, False), (2, True)):
                q = profile(
                    true_hypothesis,
                    alternative,
                    background_fraction,
                    degree,
                    response_nuisance,
                )
                rows.append(
                    {
                        "true": true_hypothesis,
                        "alternative": alternative,
                        "background_over_D_signal_exposure": background_fraction,
                        "background_polynomial_degree": degree,
                        "threshold_and_resolution_nuisance": response_nuisance,
                        "q_per_D_normalized_exposure": q,
                        "D_normalized_exposure_for_3sigma": 9 / q,
                        "D_normalized_exposure_for_5sigma": 25 / q,
                    }
                )
    assert all(row["q_per_D_normalized_exposure"] > 0 for row in rows)
    return rows


def run_all(include_profiles=True):
    """Execute the complete assertion-backed audit and return JSON-safe results."""
    result = {
        "metadata": {
            "current_convention": "gamma^mu P_L, P_L=(1-gamma5)/2",
            "metric": "+---",
            "phase_space_total": "rho/(8*pi), no identical-particle factor",
            "literature_gamma_1_minus_gamma5_factor": 4,
            "seeds": [20260810, 44117, 271828, 99, 8675309, 12345, 762341],
        },
        "gamma_and_cubature": six_axis_trace_tests(),
        "phase_space_monte_carlo": monte_carlo_phase_space(),
        "diagonal_threshold": threshold_tests(),
        "covariant_threshold": covariant_threshold_test(),
        "off_diagonal_phase": off_diagonal_tests(),
        "positivity_scan": positivity_scan(),
        "multi_kraus": kraus_tests(),
        "pair_rest_and_spread": pair_rest_tests(),
        "detector_smearing": smearing_tests(),
        "threshold_gaps": threshold_gap_tests(),
        "tomography": tomography_tests(),
        "high_boost_tradeoff": high_boost_tests(),
        "pmns_diagonal_weights": pmns_diagonal_weights(),
    }
    if include_profiles:
        result["profile_likelihood_modulation"] = profile_modulation_test()
    return py(result)


def main():
    """Parse reproducibility options, run the audit, and serialize its evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("renp_numerical_results.json"),
    )
    parser.add_argument("--skip-profiles", action="store_true")
    args = parser.parse_args()
    results = run_all(include_profiles=not args.skip_profiles)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "all assertions passed", "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
