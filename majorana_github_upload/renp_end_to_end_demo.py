#!/usr/bin/env python3
# =============================================================================
# END-TO-END MATHEMATICAL MAP: RENP PRODUCTION -> COMPUTER-SCREEN INFERENCE
# =============================================================================
#
# Purpose and epistemic scope
# ---------------------------
# This program is a deterministic, normalized design demonstration of the
# threshold-tensor observable developed in the accompanying manuscript.  It is
# NOT an absolute atomic-rate calculation, a target selection, a detector
# certification, or a calibrated discovery forecast.  The chosen signal count
# is an explicit exposure normalization.  The program asks the narrower and
# mathematically testable question: if a resolved equal-mass diagonal RENP
# branch is transferred coherently into two measured photon polarizations, does
# the complete production-to-records likelihood retain Dirac/Majorana (D/M)
# information after specified nuisance profiling?
#
# 1. Invisible-pair kinematics and exact threshold coordinate
# -------------------------------------------------------------
# After the observed photon is removed from the prepared atomic momentum, the
# invisible neutrino pair carries q^mu.  Its invariant mass is s=q_mu q^mu.
# For a resolved pair of equal mass m, define
#
#        t = (s - 4 m^2)/m^2,                 beta = sqrt(t/(4+t)),
#
# with beta=0 below threshold.  In the two-polarization frame used here,
# q/m=(q0/m,Q/m,0,0), so the projected spatial recoil is along polarization 0.
# The scan settings are values of t; negative settings are below-threshold
# controls.  MASS_EV fixes the illustrative threshold scale and Q_OVER_M fixes
# the dimensionless projected recoil.
#
# 2. Exact phase-space-integrated response tensor
# ------------------------------------------------
# Suppressing the common positive factor m^2/(8*pi), the equal-diagonal
# neutrino calculation gives a 2x2 positive response matrix
#
#        R_H(t) = beta * [ u_H(t) I_2 + v(t) (Q/m)^2 |0><0| ],
#
#        u_D(t) = (2/3)(3+t),       u_M(t) = (2/3)t,
#        v(t)   = (2/3)[1 + 2/(4+t)].
#
# Thus the recoil-orthogonal ("dark") eigenvalue scales as beta for D but as
# beta^3 for M.  At fixed beta its exact ratio is
#
#        lambda_dark,M/lambda_dark,D = 4 beta^2/(3+beta^2).
#
# R_H is an unnormalized rate/coherency matrix, not a unit-trace neutrino or
# photon density matrix.  The code explicitly sets it to zero below threshold.
# An unresolved orthogonal Kraus contribution can be injected in the low-level
# constructor for stress studies, but the packaged demonstration uses zero
# leakage and therefore tests the certified coherent-transfer case only.
#
# 3. Setting resolution and optical transfer
# -------------------------------------------
# Finite setting resolution is modeled by a normalized Gaussian convolution in
# t, followed by interpolation at every requested setting.  A calibrated,
# invertible Jones matrix J then carries the source response to the analyzer:
#
#        R'_H(t) = J R_H(t) J^dagger.
#
# J contains unequal transmission, a small rotation, and retardance.  Its
# invertibility is asserted; this is exactly the condition under which a single
# coherent transfer cannot erase the source rank distinction.
#
# 4. Polarization POVM and expected computer-level counts
# --------------------------------------------------------
# Six rank-one projectors measure H/V, D/A, and R/L.  Each complementary pair
# sums to I_2.  The nonnegative signal template is
#
#        r_H(t,p) = Tr[Pi_p R'_H(t)].
#
# It is globally normalized and assigned the explicitly selected count N_s.
# The smooth nuisance background has shared shape
#
#        b(t,p|a,c) proportional to exp[a z(t) + c z(t)^2],
#
# where z is the standardized setting and the shape is uniform across analyzer
# outcomes.  The binned means are
#
#        mu_H(t,p) = N_s r_H/sum(r_H) + N_b b(t,p),
#
# and seeded pseudo-counts obey n(t,p) ~ Poisson(mu_H(t,p)).
#
# 5. Likelihood ordering: establish a signal before asking D versus M
# --------------------------------------------------------------------
# For independent Poisson bins the factorial-free negative log likelihood is
#
#        ell(theta;H) = sum_i [mu_i(theta;H) - n_i log mu_i(theta;H)].
#
# The discovery stage profiles the background-only model H0 first, then the
# union of nonnegative D and M signals:
#
#        q0 = max{0, 2[ell_hat(H0) - min(ell_hat(D),ell_hat(M))]}.
#
# Only after H0 is rejected in a separately calibrated analysis is the
# conditional D/M statistic formed:
#
#        T_DM = 2[ell_hat(M) - ell_hat(D)],
#
# so T_DM>0 favors D and T_DM<0 favors M.  Signal and background normalizations,
# two smooth-background coefficients, threshold shift, and setting width are
# profiled within declared bounds.  Neither q0 nor T_DM is automatically a
# Gaussian significance: boundary effects, look-elsewhere structure, nuisance
# constraints, and coverage require calibration beyond this demonstration.
#
# 6. From binned outcomes to auditable detector artifacts
# --------------------------------------------------------
# The simulated chain writes five deliberately separated products:
#
#   * exposure metadata: randomized scan order, clocks, settings, and quality;
#   * calibration metadata: Jones response and analyzer/detector mapping;
#   * event records: one synthetic time-tagged pulse record per observed count;
#   * binned observed counts: the actual likelihood input; and
#   * MC truth: expected signal/background and generating label, kept separate.
#
# Event rows contain no truth label or expected-rate field.  Consistency checks
# prove that event totals reproduce every binned count, analyzer mappings are
# exact, exposure metadata are complete, and truth totals reproduce the forward
# model.  The JSON result stores fitted profiles, threshold snapshots, row
# counts, and SHA-256 hashes so all computer-screen outputs can be audited.
#
# 7. What this script does not prove
# ----------------------------------
# It does not supply the atomic matrix element, macrocoherence preparation,
# collection efficiency, target-specific systematics, empirical background
# family, or frequentist calibration needed for an experimental claim.  It
# proves an internally consistent forward/inverse pipeline under its explicit
# assumptions, exposes those assumptions in machine-readable outputs, and
# prevents D/M classification from being reported before the H0 test.
# =============================================================================

"""Synthetic discovery-then-classification demonstration for threshold-tensor RENP.

All numerical outputs are reproducible from a fixed random seed.  See the
commented mathematical map above for equations, inference ordering, and scope.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize


# Reproducibility and normalized physics benchmark.  N_SIGNAL is selected for
# the design study; it is not derived from an atomic target or running time.
SEED = 20260811
MASS_EV = 0.010
Q_OVER_M = 1.0
SIGMA_T = 0.025
N_SIGNAL = 1200.0
BACKGROUND_FRACTION = 0.10
BACKGROUND_COUNT = BACKGROUND_FRACTION * N_SIGNAL
N_PULSES_PER_EXPOSURE = 10_000
PULSE_PERIOD_NS = 1_000_000
SIGNAL_GATE_WIDTH_NS = 200
EXPOSURE_GAP_NS = 1_000_000_000
BASE_TDC_NS = 2_026_081_100_000_000_000
CALIBRATION_ID = "CAL-20260811-001"
CLOCK_CALIBRATION_ID = "CLOCK-20260811-001"
RESPONSE_KERNEL_VERSION = "equal-diagonal-PL-v2"

# A mixed scan: five closed-channel control points followed by a logarithmically
# dense set of open-channel points that resolves the nonanalytic threshold.
SETTINGS = np.r_[
    -0.60,
    -0.40,
    -0.25,
    -0.12,
    -0.04,
    np.geomspace(0.015, 3.0, 22),
]
# Three complementary polarization bases constitute an informationally
# complete qubit tomography set.  The matrices are analyzer POVM elements.
PROJECTOR_LABELS = ("H", "V", "D", "A", "R", "L")
PROJECTORS = np.array(
    [
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[0.5, 0.5], [0.5, 0.5]],
        [[0.5, -0.5], [-0.5, 0.5]],
        [[0.5, 0.5j], [-0.5j, 0.5]],
        [[0.5, -0.5j], [0.5j, 0.5]],
    ],
    dtype=complex,
)

# A deterministic, invertible optical transfer used to prove that the forward
# model can carry nontrivial loss, rotation, and retardance without changing
# the response-rank logic.  It is treated as calibration data in this demo.
_THETA = np.deg2rad(3.0)
_PHASE = 0.08
_UNITARY = np.array(
    [
        [np.cos(_THETA), -np.exp(1j * _PHASE) * np.sin(_THETA)],
        [np.sin(_THETA), np.exp(1j * _PHASE) * np.cos(_THETA)],
    ]
)
JONES = np.diag(np.sqrt([0.72, 0.68])) @ _UNITARY

# The fine latent grid supports the Gaussian convolution before interpolation
# onto the much smaller experimental scan.  It extends beyond both scan ends to
# suppress truncation artifacts from the seven-sigma filter window.
GRID_STEP = 0.001
GRID = np.arange(-1.5, 5.0 + GRID_STEP, GRID_STEP)


def to_builtin(value):
    """Recursively convert NumPy/complex objects to JSON-serializable values.

    Complex scalars are retained as explicit real/imaginary records rather
    than silently discarded; this matters for auditing the Jones calibration.
    """
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    return value


def beta_from_t(t):
    """Return the equal-mass two-body speed beta, with causal thresholding.

    Since ``s/m^2=4+t``, ``beta=sqrt(1-4m^2/s)=sqrt(t/(4+t))`` for t>=0.
    Clipping only the numerator implements the declared zero response for a
    closed channel while keeping the routine vectorized.
    """
    t = np.asarray(t)
    return np.sqrt(np.maximum(t, 0.0) / (4.0 + np.maximum(t, 0.0)))


def exact_response_on_grid(hypothesis: str, orthogonal_leakage: float = 0.0):
    """Dimensionless 2x2 response for q/m=(q0/m,Q/m,0,0).

    The common m^2/(8*pi) factor is suppressed.  With
    t=(s-4m^2)/m^2, the exact equal-diagonal coefficients are
      u_D/m^2=2(3+t)/3,
      u_M/m^2=2t/3,
      v=2[1+2/(4+t)]/3.
    The optional leakage is an explicitly labelled unresolved orthogonal
    Kraus intensity relative to the leading longitudinal term.
    """
    # Reject accidental model-name fall-through: D and M have distinct
    # threshold powers and must never share a default branch.
    if hypothesis not in ("D", "M"):
        raise ValueError("hypothesis must be D or M")
    positive = np.maximum(GRID, 0.0)
    beta = beta_from_t(GRID)

    # The isotropic coefficient u differs between D and M because identical
    # Majorana exchange cancels the threshold-constant orthogonal response.
    u = (2.0 / 3.0) * ((3.0 + positive) if hypothesis == "D" else positive)
    v = (2.0 / 3.0) * (1.0 + 2.0 / (4.0 + positive))
    q2 = Q_OVER_M**2

    # In the recoil eigenbasis, polarization 0 is bright (u+vQ^2) and
    # polarization 1 is orthogonal/dark (u).  Off-diagonal entries vanish in
    # this chosen frame before the nontrivial optical transfer is applied.
    response = np.zeros((GRID.size, 2, 2), dtype=complex)
    response[:, 0, 0] = beta * (u + v * q2)
    response[:, 1, 1] = beta * u
    # This optional term exposes the exact assumption threatened by unresolved
    # rank-adding transfer.  It is zero in RAW_RESPONSE below.
    if hypothesis == "M" and orthogonal_leakage:
        response[:, 1, 1] += beta * v * q2 * orthogonal_leakage
    response[GRID < 0] = 0.0
    return response


# Cache the unsmeared exact tensors: nuisance iterations vary only scan response,
# not the first-principles D/M coefficient functions.
RAW_RESPONSE = {hypothesis: exact_response_on_grid(hypothesis) for hypothesis in ("D", "M")}


def convolved_response(hypothesis: str, shift: float = 0.0, sigma_t: float = SIGMA_T):
    """Apply finite setting resolution and a profiled threshold offset.

    The Gaussian filter is a discrete approximation to
    ``R_obs(t)=integral dt' G_sigma(t-t') R_exact(t')``.  Interpolation at
    ``SETTINGS-shift`` implements a common calibration displacement without
    changing the exact source tensor.
    """
    raw = RAW_RESPONSE[hypothesis]
    smoothed = np.zeros_like(raw)
    # Convolve components rather than eigenvalues so the operation remains a
    # linear positive mixture of physical response matrices.
    for row in range(2):
        for column in range(2):
            smoothed[:, row, column] = gaussian_filter1d(
                raw[:, row, column].real,
                sigma_t / GRID_STEP,
                mode="constant",
                truncate=7,
            )
    at_settings = np.zeros((SETTINGS.size, 2, 2), dtype=complex)
    # The response is real in its source eigenbasis, but complex dtype is
    # retained for the subsequent Jones rotation and retardance.
    for row in range(2):
        for column in range(2):
            at_settings[:, row, column] = np.interp(
                SETTINGS - shift,
                GRID,
                smoothed[:, row, column].real,
            )
    return at_settings


def detector_template(hypothesis: str, shift: float = 0.0, sigma_t: float = SIGMA_T):
    """Map the source tensor through calibrated optics into six POVM rates.

    ``propagated[s]=J R[s] J^dagger`` and
    ``intensities[s,p]=Tr(PROJECTORS[p] propagated[s])``.  Tiny negative values
    from floating-point arithmetic are clipped; a physical positive matrix and
    positive projector have nonnegative trace overlap.
    """
    response = convolved_response(hypothesis, shift, sigma_t)
    # Index form: J_ab R_sbc (J^dagger)_cd, with ``JONES.conj()`` supplying
    # the conjugate after the explicit einsum index ordering.
    propagated = np.einsum("ab,sbc,dc->sad", JONES, response, JONES.conj())
    intensities = np.einsum("pab,sba->sp", PROJECTORS, propagated).real
    return np.maximum(intensities, 0.0), response, propagated


STANDARDIZED_SETTING = (SETTINGS - SETTINGS.mean()) / (SETTINGS.max() - SETTINGS.min())


def background_shape(linear: float, quadratic: float):
    """Return a normalized positive, smooth, analyzer-symmetric background.

    The exponential parameterization guarantees positivity.  This deliberately
    limited family is a declared demonstration assumption, not an empirical
    assertion about backgrounds in any particular RENP apparatus.
    """
    per_setting = np.exp(
        linear * STANDARDIZED_SETTING
        + quadratic * STANDARDIZED_SETTING * STANDARDIZED_SETTING
    )
    shape = np.repeat(per_setting[:, None], len(PROJECTOR_LABELS), axis=1)
    return shape / shape.sum()


TRUE_BACKGROUND_SHAPE = background_shape(0.25, -0.15)


def expectation(
    hypothesis: str,
    n_signal: float = N_SIGNAL,
    background_fraction: float = BACKGROUND_FRACTION,
):
    """Build total, signal, and background Poisson means for one hypothesis.

    Global template normalization makes ``n_signal`` the expected signal sum.
    Likewise, ``background_fraction*n_signal`` is the expected background sum.
    Absolute detection efficiency and atomic production rate are intentionally
    outside this normalized demonstration.
    """
    template, response, propagated = detector_template(hypothesis)
    signal = template / template.sum() * n_signal
    background = TRUE_BACKGROUND_SHAPE * (background_fraction * n_signal)
    return signal + background, signal, background, response, propagated


def fit_hypothesis(counts, hypothesis: str, allow_zero_signal: bool = False):
    """Profile signal/background scales, smooth background, edge, and width.

    ``allow_zero_signal`` is used for the discovery construction, where the
    signal alternative must contain the background-only model at zero signal.
    Conditional D/M fits retain the original log-signal parameterization.
    """

    def model(parameters):
        """Evaluate profiled bin means for the selected D or M signal."""
        signal_parameter, log_background, linear, quadratic, shift, log_sigma = parameters
        # Discovery needs a closed, nested signal boundary at N_s=0.  Once a
        # signal is conditionally classified, log N_s enforces strict positivity.
        signal_count = signal_parameter if allow_zero_signal else np.exp(signal_parameter)
        template, _, _ = detector_template(hypothesis, shift, np.exp(log_sigma))
        template = template / template.sum()
        return np.maximum(
            signal_count * template
            + np.exp(log_background) * background_shape(linear, quadratic),
            1e-12,
        )

    def negative_log_likelihood(parameters):
        """Factorial-free independent-Poisson negative log likelihood."""
        means = model(parameters)
        # sum log(n_i!) is independent of parameters and cancels in every
        # likelihood ratio, so omitting it improves neither nor harms the fit.
        return float(np.sum(means - counts * np.log(means)))

    # Every nuisance range is explicit.  A boundary optimum is therefore a
    # model diagnostic and cannot silently escape to an unphysical parameter.
    signal_bounds = (0.0, 2.0e4) if allow_zero_signal else (np.log(20.0), np.log(2.0e4))
    bounds = [
        signal_bounds,
        (np.log(0.1), np.log(2.0e4)),
        (-2.0, 2.0),
        (-2.0, 2.0),
        (-0.025, 0.025),
        (np.log(0.015), np.log(0.040)),
    ]
    best = None
    # Multistart initialization reduces local-minimum risk near the N_s=0
    # boundary and near either end of the allowed threshold-shift interval.
    if allow_zero_signal:
        signal_starts = (0.0, 20.0, max(np.sum(counts) / 1.1, 20.0))
    else:
        signal_starts = (np.log(max(np.sum(counts) / 1.1, 20.0)),)
    for signal_start in signal_starts:
        for initial_shift in (-0.020, 0.0, 0.020):
            initial = np.array(
                [
                    signal_start,
                    np.log(max(np.sum(counts) / 11.0, 0.1)),
                    0.0,
                    0.0,
                    initial_shift,
                    np.log(SIGMA_T),
                ]
            )
            fit = minimize(
                negative_log_likelihood,
                initial,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 1200, "ftol": 1e-12, "gtol": 1e-8},
            )
            if best is None or fit.fun < best.fun:
                best = fit
    if best is None:
        raise RuntimeError("profile fit did not run")
    return {
        "nll_without_factorial": float(best.fun),
        "success": bool(best.success),
        "message": str(best.message),
        "signal_count": float(best.x[0] if allow_zero_signal else np.exp(best.x[0])),
        "background_count": float(np.exp(best.x[1])),
        "background_linear": float(best.x[2]),
        "background_quadratic": float(best.x[3]),
        "threshold_shift_t": float(best.x[4]),
        "sigma_t": float(np.exp(best.x[5])),
        "model": model(best.x),
    }


def fit_background_only(counts):
    """Fit the bounded H0 profile using the identical background family.

    Sharing the background parameterization is essential: otherwise q0 could
    measure a mismatch between unrelated nuisance models rather than evidence
    for a threshold signal.
    """

    def model(parameters):
        """Evaluate the H0 bin means; by construction the signal is absent."""
        log_background, linear, quadratic = parameters
        return np.maximum(
            np.exp(log_background) * background_shape(linear, quadratic),
            1e-12,
        )

    def negative_log_likelihood(parameters):
        """Factorial-free independent-Poisson H0 objective."""
        means = model(parameters)
        return float(np.sum(means - counts * np.log(means)))

    bounds = [
        (np.log(0.1), np.log(2.0e4)),
        (-2.0, 2.0),
        (-2.0, 2.0),
    ]
    best = None
    # Start the slope from both signs and zero as a compact convexity check for
    # the bounded smooth-background profile.
    for initial_linear in (-1.0, 0.0, 1.0):
        initial = np.array(
            [
                np.log(max(float(np.sum(counts)), 0.1)),
                initial_linear,
                0.0,
            ]
        )
        fit = minimize(
            negative_log_likelihood,
            initial,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1200, "ftol": 1e-12, "gtol": 1e-8},
        )
        if best is None or fit.fun < best.fun:
            best = fit
    if best is None:
        raise RuntimeError("background-only profile fit did not run")
    return {
        "nll_without_factorial": float(best.fun),
        "success": bool(best.success),
        "message": str(best.message),
        "signal_count": 0.0,
        "background_count": float(np.exp(best.x[0])),
        "background_linear": float(best.x[1]),
        "background_quadratic": float(best.x[2]),
        "model": model(best.x),
    }


def discovery_profile(counts, signal_fits=None):
    """Profile H0 first, then the union of nonnegative D and M signals.

    The alternative is a union rather than a preselected physics hypothesis.
    The statistic is clipped at zero because a discovery likelihood ratio must
    not become negative through optimizer tolerance.  Its null distribution is
    not asserted to be chi-square in this program.
    """

    if signal_fits is None:
        signal_fits = {
            hypothesis: fit_hypothesis(counts, hypothesis, allow_zero_signal=True)
            for hypothesis in ("D", "M")
        }
    h0_fit = fit_background_only(counts)
    # The signal-union likelihood is the better of the separately profiled D
    # and M alternatives; this avoids classifying an unestablished excess.
    best_hypothesis = min(signal_fits, key=lambda hypothesis: signal_fits[hypothesis]["nll_without_factorial"])
    best_signal_fit = signal_fits[best_hypothesis]
    q0 = max(
        0.0,
        2.0 * (h0_fit["nll_without_factorial"] - best_signal_fit["nll_without_factorial"]),
    )
    return {
        "q0_background_only_vs_signal_union": q0,
        "best_signal_hypothesis": best_hypothesis,
        "background_only_fit": h0_fit,
        "signal_union_fits": signal_fits,
    }


def response_snapshot(t: float):
    """Return an analytic above-threshold tensor check at one scan point.

    Both the direct s,m expression and its beta reparameterization are emitted
    for the dark-eigenvalue M/D ratio.  Their equality is later asserted to
    machine precision, independently of the convolution and fit pipeline.
    """
    if t <= 0:
        raise ValueError("snapshot must be above threshold")
    beta = float(beta_from_t(t))
    s_eV2 = MASS_EV * MASS_EV * (4.0 + t)
    q_eV = Q_OVER_M * MASS_EV
    output = {
        "t": t,
        "beta": beta,
        "s_eV2": s_eV2,
        "delta_eV2": MASS_EV * MASS_EV * t,
        "Q_eV": q_eV,
        "hypotheses": {},
    }
    for hypothesis in ("D", "M"):
        positive = t
        u = (2.0 / 3.0) * ((3.0 + positive) if hypothesis == "D" else positive)
        v = (2.0 / 3.0) * (1.0 + 2.0 / (4.0 + positive))
        # Ordered as bright (recoil-supported) then dark (recoil-orthogonal).
        eigenvalues = beta * np.array([u + v * Q_OVER_M**2, u])
        output["hypotheses"][hypothesis] = {
            "dimensionless_response_eigenvalues_bright_dark": eigenvalues,
            "bright_probability_conditional_on_signal": eigenvalues[0] / eigenvalues.sum(),
            "dark_probability_conditional_on_signal": eigenvalues[1] / eigenvalues.sum(),
        }
    output["exact_dark_M_over_D"] = (s_eV2 - 4 * MASS_EV**2) / (s_eV2 - MASS_EV**2)
    output["exact_dark_M_over_D_beta_form"] = 4 * beta**2 / (3 + beta**2)
    return output


def strip_model(fit):
    """Remove a dense NumPy bin array before JSON serialization of fit data."""
    return {key: value for key, value in fit.items() if key != "model"}


def build_exposure_schedule(dataset_data):
    """Create opaque, randomized exposure metadata independently of MC truth.

    Randomizing acquisition order breaks a simple correlation between setting
    and laboratory drift.  The mapping is reproducible but carries no D/M label
    into the detector-facing exposure table.
    """

    schedule_rng = np.random.default_rng(SEED + 101)
    duration_ns = N_PULSES_PER_EXPOSURE * PULSE_PERIOD_NS
    schedules = {}
    exposure_rows = []
    for dataset_index, dataset in enumerate(dataset_data):
        acquisition_order = schedule_rng.permutation(len(SETTINGS))
        for acquisition_index, setting_index in enumerate(acquisition_order):
            exposure_id = f"EXP-{dataset_index + 1:04d}-{setting_index + 1:04d}"
            start_ns = (
                BASE_TDC_NS
                + dataset_index * 1_000_000_000_000
                + acquisition_index * (duration_ns + EXPOSURE_GAP_NS)
            )
            stop_ns = start_ns + duration_ns
            t = float(SETTINGS[setting_index])
            # In a real system these identifiers and timing fields would be
            # populated by acquisition services; here their schema is tested.
            row = {
                "dataset_id": dataset["dataset_id"],
                "exposure_id": exposure_id,
                "acquisition_index": acquisition_index,
                "setting_index": int(setting_index),
                "t_setting": f"{t:.12g}",
                "s_eV2": f"{MASS_EV**2 * (4 + t):.12g}",
                "beta_open_channel": f"{float(beta_from_t(t)):.12g}",
                "channel_open": int(t >= 0),
                "start_tdc_ns": start_ns,
                "stop_tdc_ns": stop_ns,
                "pulse_count": N_PULSES_PER_EXPOSURE,
                "pulse_period_ns": PULSE_PERIOD_NS,
                "signal_gate_width_ns": SIGNAL_GATE_WIDTH_NS,
                "live_time_s": f"{duration_ns * 1e-9:.9g}",
                "cycle_type": "BELOW_THRESHOLD_CONTROL" if t < 0 else "PHYSICS_CANDIDATE",
                "calibration_id": CALIBRATION_ID,
                "clock_calibration_id": CLOCK_CALIBRATION_ID,
                "response_kernel_version": RESPONSE_KERNEL_VERSION,
                "randomization_key": "ORDER-20260811-001",
                "quality_mask": "PASS",
            }
            schedules[(dataset["dataset_id"], int(setting_index))] = row
            exposure_rows.append(row)
    return schedules, exposure_rows


def write_calibration_table(path: Path):
    """Write the declared detector/analyzer/Jones calibration snapshot.

    Each detector corresponds to one POVM outcome.  Repeating J in every row is
    intentionally denormalized so a standalone detector row remains traceable
    to the optical response version used by the forward model.
    """
    fields = [
        "calibration_id",
        "detector_id",
        "analyzer_projector",
        "basis",
        "outcome",
        "modeled_relative_efficiency",
        "modeled_dark_rate_hz",
        "modeled_dead_time_ns",
        "modeled_crosstalk_probability",
        "j00_real",
        "j00_imag",
        "j01_real",
        "j01_imag",
        "j10_real",
        "j10_imag",
        "j11_real",
        "j11_imag",
        "response_kernel_version",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for projector_index, label in enumerate(PROJECTOR_LABELS):
            # Adjacent rows are the +/- outcomes of Z, X, and Y respectively.
            basis = ("Z", "Z", "X", "X", "Y", "Y")[projector_index]
            outcome = "+" if projector_index % 2 == 0 else "-"
            writer.writerow(
                {
                    "calibration_id": CALIBRATION_ID,
                    "detector_id": f"SNSPD-{projector_index + 1:02d}",
                    "analyzer_projector": label,
                    "basis": basis,
                    "outcome": outcome,
                    "modeled_relative_efficiency": "1",
                    "modeled_dark_rate_hz": "0",
                    "modeled_dead_time_ns": "0",
                    "modeled_crosstalk_probability": "0",
                    "j00_real": f"{JONES[0, 0].real:.16g}",
                    "j00_imag": f"{JONES[0, 0].imag:.16g}",
                    "j01_real": f"{JONES[0, 1].real:.16g}",
                    "j01_imag": f"{JONES[0, 1].imag:.16g}",
                    "j10_real": f"{JONES[1, 0].real:.16g}",
                    "j10_imag": f"{JONES[1, 0].imag:.16g}",
                    "j11_real": f"{JONES[1, 1].real:.16g}",
                    "j11_imag": f"{JONES[1, 1].imag:.16g}",
                    "response_kernel_version": RESPONSE_KERNEL_VERSION,
                }
            )


def write_synthetic_tables(
    dataset_data,
    binned_path: Path,
    events_path: Path,
    exposures_path: Path,
    calibrations_path: Path,
    mc_truth_path: Path,
):
    """Write observations separately from truth and instrument metadata.

    This separation models an essential blindness rule: the inference-facing
    event and binned tables contain measured-like fields only, while generator
    labels and expected components live exclusively in ``mc_truth_path``.
    """

    schedules, exposure_rows = build_exposure_schedule(dataset_data)
    # The exposure and calibration tables are written first because every
    # observed row carries foreign-key-like identifiers into these snapshots.
    with exposures_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exposure_rows[0]))
        writer.writeheader()
        writer.writerows(exposure_rows)

    write_calibration_table(calibrations_path)

    binned_fields = [
        "bin_id",
        "dataset_id",
        "exposure_id",
        "t_setting",
        "s_eV2",
        "beta_open_channel",
        "channel_open",
        "analyzer_projector",
        "observed_count",
        "calibration_id",
        "response_kernel_version",
        "quality_mask",
    ]
    truth_fields = [
        "bin_id",
        "dataset_id",
        "true_hypothesis",
        "poisson_seed",
        "expected_signal",
        "expected_background",
        "expected_total",
    ]
    binned_rows = []
    truth_rows = []
    # Exactly one binned row and one separate truth row are produced for every
    # dataset x setting x analyzer cell.
    for dataset_index, dataset in enumerate(dataset_data):
        for setting_index, t in enumerate(SETTINGS):
            exposure = schedules[(dataset["dataset_id"], setting_index)]
            for projector_index, label in enumerate(PROJECTOR_LABELS):
                bin_id = f"BIN-{dataset_index + 1:04d}-{setting_index + 1:04d}-{projector_index + 1:02d}"
                binned_rows.append(
                    {
                        "bin_id": bin_id,
                        "dataset_id": dataset["dataset_id"],
                        "exposure_id": exposure["exposure_id"],
                        "t_setting": f"{t:.12g}",
                        "s_eV2": f"{MASS_EV**2 * (4 + t):.12g}",
                        "beta_open_channel": f"{float(beta_from_t(t)):.12g}",
                        "channel_open": int(t >= 0),
                        "analyzer_projector": label,
                        "observed_count": int(dataset["observed"][setting_index, projector_index]),
                        "calibration_id": CALIBRATION_ID,
                        "response_kernel_version": RESPONSE_KERNEL_VERSION,
                        "quality_mask": "PASS",
                    }
                )
                # MC-only quantities never enter ``binned_rows``.
                truth_rows.append(
                    {
                        "bin_id": bin_id,
                        "dataset_id": dataset["dataset_id"],
                        "true_hypothesis": dataset["true_hypothesis"],
                        "poisson_seed": SEED,
                        "expected_signal": f"{dataset['signal'][setting_index, projector_index]:.12g}",
                        "expected_background": f"{dataset['background'][setting_index, projector_index]:.12g}",
                        "expected_total": f"{dataset['expected'][setting_index, projector_index]:.12g}",
                    }
                )
    with binned_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=binned_fields)
        writer.writeheader()
        writer.writerows(binned_rows)
    with mc_truth_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=truth_fields)
        writer.writeheader()
        writer.writerows(truth_rows)

    event_fields = [
        "event_id",
        "dataset_id",
        "tdc_time_ns",
        "pulse_id",
        "exposure_id",
        "detector_id",
        "basis",
        "outcome",
        "pulse_height_mV",
        "time_over_threshold_ns",
        "gate_code",
        "veto_bits",
        "calibration_id",
        "clock_calibration_id",
        "quality_mask",
    ]
    # Event pulse morphology is auxiliary and uses a seed independent from the
    # Poisson-count generator and scan-order generator.
    event_rng = np.random.default_rng(SEED + 202)
    pending_events = []
    for dataset in dataset_data:
        for setting_index, _ in enumerate(SETTINGS):
            exposure = schedules[(dataset["dataset_id"], setting_index)]
            start_ns = int(exposure["start_tdc_ns"])
            for projector_index, label in enumerate(PROJECTOR_LABELS):
                count = int(dataset["observed"][setting_index, projector_index])
                basis = ("Z", "Z", "X", "X", "Y", "Y")[projector_index]
                outcome = "+" if projector_index % 2 == 0 else "-"
                # Conditional on the already generated bin count, distribute
                # records over pulses and inside the signal gate.  This creates
                # genuine row-per-pulse artifacts without claiming a physical
                # point-process or detector-pulse model.
                pulse_indices = event_rng.integers(0, N_PULSES_PER_EXPOSURE, size=count)
                gate_offsets = event_rng.integers(0, SIGNAL_GATE_WIDTH_NS, size=count)
                heights = np.maximum(event_rng.normal(32.0, 2.0, size=count), 1.0)
                widths = np.maximum(event_rng.normal(8.0, 0.6, size=count), 0.5)
                for pulse_index, gate_offset, height, width in zip(
                    pulse_indices, gate_offsets, heights, widths
                ):
                    tdc_time_ns = start_ns + int(pulse_index) * PULSE_PERIOD_NS + int(gate_offset)
                    pending_events.append(
                        {
                            "dataset_id": dataset["dataset_id"],
                            "tdc_time_ns": tdc_time_ns,
                            "pulse_id": f"PULSE-{exposure['exposure_id']}-{int(pulse_index):05d}",
                            "exposure_id": exposure["exposure_id"],
                            "detector_id": f"SNSPD-{projector_index + 1:02d}",
                            "basis": basis,
                            "outcome": outcome,
                            "pulse_height_mV": f"{height:.6f}",
                            "time_over_threshold_ns": f"{width:.6f}",
                            "gate_code": "SIGNAL",
                            "veto_bits": "0x0000",
                            "calibration_id": CALIBRATION_ID,
                            "clock_calibration_id": CLOCK_CALIBRATION_ID,
                            "quality_mask": "PASS",
                        }
                    )
    # Time ordering precedes stable event-ID assignment, mirroring an immutable
    # acquisition stream rather than grouping events by the simulated truth.
    pending_events.sort(key=lambda row: (row["tdc_time_ns"], row["dataset_id"], row["detector_id"]))
    for event_index, row in enumerate(pending_events, start=1):
        row["event_id"] = f"EVT-{event_index:012d}"
    with events_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=event_fields)
        writer.writeheader()
        writer.writerows(pending_events)

    return {
        "binned_rows": binned_rows,
        "truth_rows": truth_rows,
        "exposure_rows": exposure_rows,
        "event_rows": pending_events,
    }


def file_sha256(path: Path):
    """Return the content hash recorded in the machine-readable manifest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_demo(
    output_json: Path,
    binned_path: Path,
    events_path: Path,
    exposures_path: Path,
    calibrations_path: Path,
    mc_truth_path: Path,
):
    """Execute the complete forward simulation, inference, and audit chain.

    The return value is the same object written to JSON.  CSV outputs are
    created before hashes are calculated, and all mathematical/schema
    assertions must pass before the result file is committed.
    """
    # This RNG controls Poisson pseudo-counts only.  Separate deterministic RNGs
    # inside table writers isolate acquisition-order and pulse-shape draws.
    rng = np.random.default_rng(SEED)

    # Select t corresponding exactly to beta=0.15 for the analytic tensor
    # snapshot: t=4 beta^2/(1-beta^2).
    snapshot_t = 4 * 0.15**2 / (1 - 0.15**2)
    snapshot = response_snapshot(snapshot_t)

    results = {
        "metadata": {
            "purpose": (
                "normalized discovery-then-conditional-D/M demonstration; "
                "not a target-specific rate prediction or calibrated significance"
            ),
            "seed": SEED,
            "mass_eV": MASS_EV,
            "Q_over_m": Q_OVER_M,
            "scan_variable": "t=(s-4m^2)/m^2",
            "gaussian_sigma_t": SIGMA_T,
            "selected_signal_counts_per_demonstration": N_SIGNAL,
            "background_over_signal": BACKGROUND_FRACTION,
            "projectors": list(PROJECTOR_LABELS),
            "jones_matrix": to_builtin(JONES),
            "hypothesis_order": [
                "first profile H0 background-only against the D-or-M signal union",
                "only then conditionally compare D against M",
            ],
            "instrument_scope": (
                "binned Poisson demonstration with fixed Jones/POVM calibration; "
                "event-level artifacts are synthetic records, not a full time-process likelihood"
            ),
        },
        "physics_snapshot_beta_0p15": to_builtin(snapshot),
        "discovery_profiles": {
            "asimov_signal_records": {},
            "seeded_signal_records": {},
            "background_only_control": {},
        },
        "asimov_profiles": {},
        "seeded_pseudodata": {},
    }

    dataset_data = []
    all_fits = []
    # Generate one signal-bearing design dataset from each hypothesis.  Each is
    # analyzed under both D and M; no generating label enters either fit.
    for dataset_index, true_hypothesis in enumerate(("D", "M"), start=1):
        expected, signal, background, _, _ = expectation(true_hypothesis)

        # Asimov data equal the exact bin means and expose median-separation
        # behavior without a Poisson realization.
        asimov_fits = {hypothesis: fit_hypothesis(expected, hypothesis) for hypothesis in ("D", "M")}
        all_fits.extend(asimov_fits.values())
        asimov_discovery = discovery_profile(expected)
        all_fits.append(asimov_discovery["background_only_fit"])
        all_fits.extend(asimov_discovery["signal_union_fits"].values())
        # Sign convention is fixed globally: positive favors D, negative M.
        signed_asimov_q = 2 * (
            asimov_fits["M"]["nll_without_factorial"]
            - asimov_fits["D"]["nll_without_factorial"]
        )
        correct_q = signed_asimov_q if true_hypothesis == "D" else -signed_asimov_q
        results["asimov_profiles"][true_hypothesis] = {
            "signed_q_positive_favors_D": signed_asimov_q,
            "q_against_wrong_hypothesis": correct_q,
            "equivalent_sqrt_q_sigma_only_if_wilks_regular": np.sqrt(max(correct_q, 0.0)),
            "selected_signal_counts_for_q25_by_linear_exposure_scaling": N_SIGNAL * 25 / correct_q,
            "fits": {
                hypothesis: {key: value for key, value in fit.items() if key != "model"}
                for hypothesis, fit in asimov_fits.items()
            },
        }
        results["discovery_profiles"]["asimov_signal_records"][true_hypothesis] = {
            "q0_background_only_vs_signal_union": asimov_discovery[
                "q0_background_only_vs_signal_union"
            ],
            "best_signal_hypothesis": asimov_discovery["best_signal_hypothesis"],
            "background_only_fit": strip_model(asimov_discovery["background_only_fit"]),
            "signal_union_fits": {
                hypothesis: strip_model(fit)
                for hypothesis, fit in asimov_discovery["signal_union_fits"].items()
            },
        }

        # Seeded pseudo-data exercise the same inference on integer observations.
        observed = rng.poisson(expected)
        observed_fits = {hypothesis: fit_hypothesis(observed, hypothesis) for hypothesis in ("D", "M")}
        all_fits.extend(observed_fits.values())
        observed_discovery = discovery_profile(observed)
        all_fits.append(observed_discovery["background_only_fit"])
        all_fits.extend(observed_discovery["signal_union_fits"].values())
        signed_observed_q = 2 * (
            observed_fits["M"]["nll_without_factorial"]
            - observed_fits["D"]["nll_without_factorial"]
        )
        results["seeded_pseudodata"][true_hypothesis] = {
            "total_observed": int(observed.sum()),
            "signed_q_positive_favors_D": signed_observed_q,
            "fits": {
                hypothesis: {key: value for key, value in fit.items() if key != "model"}
                for hypothesis, fit in observed_fits.items()
            },
            "first_twelve_binned_counts": [
                {
                    "t": SETTINGS[index // len(PROJECTOR_LABELS)],
                    "projector": PROJECTOR_LABELS[index % len(PROJECTOR_LABELS)],
                    "expected": expected.reshape(-1)[index],
                    "observed": observed.reshape(-1)[index],
                }
                for index in range(12)
            ],
        }
        results["discovery_profiles"]["seeded_signal_records"][true_hypothesis] = {
            "q0_background_only_vs_signal_union": observed_discovery[
                "q0_background_only_vs_signal_union"
            ],
            "best_signal_hypothesis": observed_discovery["best_signal_hypothesis"],
            "background_only_fit": strip_model(observed_discovery["background_only_fit"]),
            "signal_union_fits": {
                hypothesis: strip_model(fit)
                for hypothesis, fit in observed_discovery["signal_union_fits"].items()
            },
        }
        dataset_data.append(
            {
                "dataset_id": f"SIM-{dataset_index:04d}",
                "true_hypothesis": true_hypothesis,
                "expected": expected,
                "signal": signal,
                "background": background,
                "observed": observed,
            }
        )

    # An explicit background-only control is indispensable: it verifies that
    # the nested signal fit can remain on N_s=0 rather than manufacture a
    # threshold feature from the permitted smooth background.
    h0_signal = np.zeros_like(TRUE_BACKGROUND_SHAPE)
    h0_background = TRUE_BACKGROUND_SHAPE * BACKGROUND_COUNT
    h0_expected = h0_signal + h0_background
    h0_asimov_discovery = discovery_profile(h0_expected)
    all_fits.append(h0_asimov_discovery["background_only_fit"])
    all_fits.extend(h0_asimov_discovery["signal_union_fits"].values())
    h0_observed = rng.poisson(h0_expected)
    h0_seeded_discovery = discovery_profile(h0_observed)
    all_fits.append(h0_seeded_discovery["background_only_fit"])
    all_fits.extend(h0_seeded_discovery["signal_union_fits"].values())
    results["discovery_profiles"]["background_only_control"] = {
        "asimov": {
            "total_expected": float(h0_expected.sum()),
            "q0_background_only_vs_signal_union": h0_asimov_discovery[
                "q0_background_only_vs_signal_union"
            ],
            "best_signal_hypothesis": h0_asimov_discovery["best_signal_hypothesis"],
            "background_only_fit": strip_model(h0_asimov_discovery["background_only_fit"]),
            "signal_union_fits": {
                hypothesis: strip_model(fit)
                for hypothesis, fit in h0_asimov_discovery["signal_union_fits"].items()
            },
        },
        "seeded": {
            "total_observed": int(h0_observed.sum()),
            "q0_background_only_vs_signal_union": h0_seeded_discovery[
                "q0_background_only_vs_signal_union"
            ],
            "best_signal_hypothesis": h0_seeded_discovery["best_signal_hypothesis"],
            "background_only_fit": strip_model(h0_seeded_discovery["background_only_fit"]),
            "signal_union_fits": {
                hypothesis: strip_model(fit)
                for hypothesis, fit in h0_seeded_discovery["signal_union_fits"].items()
            },
        },
    }
    dataset_data.append(
        {
            "dataset_id": "SIM-0003",
            "true_hypothesis": "H0",
            "expected": h0_expected,
            "signal": h0_signal,
            "background": h0_background,
            "observed": h0_observed,
        }
    )

    # Materialize detector-facing records only after all three logical datasets
    # (D signal, M signal, and H0 control) have been constructed.
    tables = write_synthetic_tables(
        dataset_data,
        binned_path,
        events_path,
        exposures_path,
        calibrations_path,
        mc_truth_path,
    )

    # ---------------------------------------------------------------------
    # Deterministic consistency assertions spanning every link in the demo.
    # ---------------------------------------------------------------------
    # POVM validity: complementary outcomes sum to identity and every element
    # is Hermitian, unit trace, and positive semidefinite.
    for first, second in ((0, 1), (2, 3), (4, 5)):
        assert np.allclose(PROJECTORS[first] + PROJECTORS[second], np.eye(2))
    for projector in PROJECTORS:
        assert np.allclose(projector, projector.conj().T)
        assert abs(np.trace(projector).real - 1.0) < 1e-14
        assert np.min(np.linalg.eigvalsh(projector)) > -1e-14
    # Optical transfer must be full rank, and the two analytic formulas for the
    # recoil-dark ratio must agree independently of the numerical scan.
    assert np.linalg.matrix_rank(JONES) == 2
    assert abs(snapshot["exact_dark_M_over_D"] - snapshot["exact_dark_M_over_D_beta_form"]) < 1e-14
    # Every nuisance profile must report convergence before statistics are used.
    assert all(fit["success"] for fit in all_fits)

    # These are regression thresholds for the selected normalized design, not
    # claims of calibrated 5-sigma discovery or classification.
    assert results["discovery_profiles"]["asimov_signal_records"]["D"][
        "q0_background_only_vs_signal_union"
    ] > 25
    assert results["discovery_profiles"]["asimov_signal_records"]["M"][
        "q0_background_only_vs_signal_union"
    ] > 25
    assert results["discovery_profiles"]["background_only_control"]["asimov"][
        "q0_background_only_vs_signal_union"
    ] < 1e-6
    assert results["asimov_profiles"]["D"]["q_against_wrong_hypothesis"] > 25
    assert results["asimov_profiles"]["M"]["q_against_wrong_hypothesis"] > 25
    assert results["seeded_pseudodata"]["D"]["signed_q_positive_favors_D"] > 0
    assert results["seeded_pseudodata"]["M"]["signed_q_positive_favors_D"] < 0

    # Schema/cardinality checks ensure a complete rectangular likelihood table
    # and one independently stored truth row per observed bin.
    expected_binned_rows = len(dataset_data) * len(SETTINGS) * len(PROJECTOR_LABELS)
    assert len(tables["binned_rows"]) == expected_binned_rows
    assert len(tables["truth_rows"]) == expected_binned_rows
    assert len(tables["exposure_rows"]) == len(dataset_data) * len(SETTINGS)
    assert len(tables["event_rows"]) == sum(int(dataset["observed"].sum()) for dataset in dataset_data)
    assert len({row["event_id"] for row in tables["event_rows"]}) == len(tables["event_rows"])
    assert [row["tdc_time_ns"] for row in tables["event_rows"]] == sorted(
        row["tdc_time_ns"] for row in tables["event_rows"]
    )
    # Enforce blindness structurally: no generator hypothesis or expected
    # component is allowed in an event record.
    forbidden_event_fields = {
        "true_hypothesis",
        "true_hypothesis_for_simulation",
        "expected_signal",
        "expected_background",
        "expected_total",
    }
    assert forbidden_event_fields.isdisjoint(tables["event_rows"][0])
    # Re-aggregate the event stream and demand exact equality with every binned
    # detector count.  This is the computer-record end of the concept chain.
    event_bin_counts = {}
    for row in tables["event_rows"]:
        key = (row["dataset_id"], row["exposure_id"], row["detector_id"])
        event_bin_counts[key] = event_bin_counts.get(key, 0) + 1
    detector_for_projector = {
        label: f"SNSPD-{index + 1:02d}" for index, label in enumerate(PROJECTOR_LABELS)
    }
    for row in tables["binned_rows"]:
        key = (
            row["dataset_id"],
            row["exposure_id"],
            detector_for_projector[row["analyzer_projector"]],
        )
        assert event_bin_counts.get(key, 0) == int(row["observed_count"])
    # Dataset-level totals independently close event -> bin -> forward-model
    # accounting for both observed counts and expected MC components.
    for dataset in dataset_data:
        dataset_id = dataset["dataset_id"]
        event_total = sum(row["dataset_id"] == dataset_id for row in tables["event_rows"])
        binned_total = sum(
            int(row["observed_count"])
            for row in tables["binned_rows"]
            if row["dataset_id"] == dataset_id
        )
        assert event_total == binned_total == int(dataset["observed"].sum())
        truth_signal = sum(
            float(row["expected_signal"])
            for row in tables["truth_rows"]
            if row["dataset_id"] == dataset_id
        )
        truth_background = sum(
            float(row["expected_background"])
            for row in tables["truth_rows"]
            if row["dataset_id"] == dataset_id
        )
        assert abs(truth_signal - float(dataset["signal"].sum())) < 1e-8
        assert abs(truth_background - float(dataset["background"].sum())) < 1e-8

    # Hash every external data artifact after all validations.  The hashes make
    # accidental post-generation edits detectable without mixing truth into data.
    artifact_paths = {
        "binned_counts": binned_path,
        "event_records": events_path,
        "exposures": exposures_path,
        "calibrations": calibrations_path,
        "mc_truth": mc_truth_path,
    }
    results["artifact_manifest"] = {
        key: {
            "path": path.name,
            "sha256": file_sha256(path),
        }
        for key, path in artifact_paths.items()
    }
    results["artifact_manifest"]["row_counts"] = {
        "binned_counts": len(tables["binned_rows"]),
        "event_records": len(tables["event_rows"]),
        "exposures": len(tables["exposure_rows"]),
        "calibrations": len(PROJECTOR_LABELS),
        "mc_truth": len(tables["truth_rows"]),
    }

    # JSON is written last, so its manifest can describe the finalized CSVs.
    output_json.write_text(json.dumps(to_builtin(results), indent=2) + "\n")
    return results


def main():
    """Parse output locations, run the audited pipeline, and print a digest."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(__file__).with_name("renp_end_to_end_results.json"),
    )
    parser.add_argument(
        "--output-binned-csv",
        type=Path,
        default=Path(__file__).with_name("renp_synthetic_binned_counts.csv"),
    )
    parser.add_argument(
        "--output-events-csv",
        type=Path,
        default=Path(__file__).with_name("renp_synthetic_events.csv"),
    )
    parser.add_argument(
        "--output-exposures-csv",
        type=Path,
        default=Path(__file__).with_name("renp_synthetic_exposures.csv"),
    )
    parser.add_argument(
        "--output-calibrations-csv",
        type=Path,
        default=Path(__file__).with_name("renp_synthetic_calibrations.csv"),
    )
    parser.add_argument(
        "--output-mc-truth-csv",
        type=Path,
        default=Path(__file__).with_name("renp_synthetic_mc_truth.csv"),
    )
    args = parser.parse_args()
    results = run_demo(
        args.output_json,
        args.output_binned_csv,
        args.output_events_csv,
        args.output_exposures_csv,
        args.output_calibrations_csv,
        args.output_mc_truth_csv,
    )
    # Keep stdout compact and automation-friendly; detailed matrices and fits
    # remain in the JSON output rather than being rounded in terminal text.
    print(
        json.dumps(
            {
                "status": "all assertions passed",
                "output_json": str(args.output_json),
                "artifact_manifest": results["artifact_manifest"],
                "asimov_discovery_q0": {
                    key: value["q0_background_only_vs_signal_union"]
                    for key, value in results["discovery_profiles"][
                        "asimov_signal_records"
                    ].items()
                },
                "seeded_discovery_q0": {
                    key: value["q0_background_only_vs_signal_union"]
                    for key, value in results["discovery_profiles"][
                        "seeded_signal_records"
                    ].items()
                },
                "asimov_q": {
                    key: value["q_against_wrong_hypothesis"]
                    for key, value in results["asimov_profiles"].items()
                },
                "seeded_signed_q_positive_favors_D": {
                    key: value["signed_q_positive_favors_D"]
                    for key, value in results["seeded_pseudodata"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
