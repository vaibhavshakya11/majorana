#!/usr/bin/env python3
"""Independent deterministic audit of the RENP threshold-tensor mathematics.

MATHEMATICAL MAP
================
This file is intentionally a compact, independently executable cross-check of
the derivation in the accompanying paper.  It does not import the production
simulation or detector model.  The calculation proceeds as follows.

1. Conventions and spin sums
----------------------------
We use natural units, the metric

    g^{mu nu} = diag(+1, -1, -1, -1),

the Dirac representation of the gamma matrices, ``slash(p) = gamma^mu p_mu``,
and ``P_L=(1-gamma^5)/2``.  For an emitted neutrino--antineutrino pair with
``q=p1+p2`` and ``s=q^2``, the direct left-handed spin sum is evaluated from

    L_D^{mu nu} = Tr[(slash(p1)+m1) gamma^mu P_L
                     (slash(p2)-m2) gamma^nu P_L].

The crossed left/right trace, which supplies the Majorana exchange term, is
checked directly against the representation-independent identity

    L_X^{mu nu} = -2 m1 m2 g^{mu nu}.

2. Exact two-body phase-space integration
------------------------------------------
With

    beta_12 = sqrt([1-(m1+m2)^2/s][1-(m1-m2)^2/s]),
    integral dPhi_2 = beta_12/(8 pi),

Lorentz covariance restricts the integrated tensor to ``g^{mu nu}`` and
``q^mu q^nu``.  The closed forms audited below are

    D^{mu nu} = beta/(24 pi) [c_g g^{mu nu} + c_q q^mu q^nu],
    c_g = s {beta^2 - 3[1-(m1^2+m2^2)/s]},
    c_q = 2 {1+(m1^2+m2^2)/s
               -2(m1^2-m2^2)^2/s^2},

and

    X^{mu nu} = -beta m1 m2 g^{mu nu}/(4 pi).

In the ordered-pair convention used here, including the uniform Majorana
factor of one half, a channel with weak coefficient ``a`` is represented by

    W_M = |a|^2 D - Re(a^2) X.

For an equal-mass diagonal channel, ``a`` is real in the relevant convention,
and this reduces in the code to ``W_M=D-X``; the matched Dirac hypothesis is
``W_D=D``.  A seeded angular Monte Carlo integrates the unintegrated gamma
traces in a boosted frame and compares them with these covariant formulas.

3. Detector-facing response and the conditional threshold theorem
-------------------------------------------------------------------
An arbitrary fixed two-output transfer/current matrix ``J_A^mu`` maps the
neutrino tensor into the Hermitian polarization response

    R_AB = J_A,mu W^{mu nu} J_B,nu^*,

where lowering the current index is essential with the chosen metric.  Let
``delta=s-4m^2`` and ``beta ~ sqrt(delta)`` at a resolved equal-mass diagonal
threshold.  For one coherent rank-supporting transfer channel, the leading
Dirac coefficient is generically rank two, whereas the leading Majorana
coefficient is the rank-one outer product of ``c_A=J_A,mu q^mu``.  Therefore

    lambda_min(R_D) = O(beta)   = O(delta^(1/2)),
    lambda_min(R_M) = O(beta^3) = O(delta^(3/2)).

Projecting onto the analyzer ray orthogonal to ``c`` gives the exact ratio

    R_M,dark/R_D,dark = (s-4m^2)/(s-m^2)
                       = 4 beta^2/(3+beta^2).

The theorem is deliberately conditional: the threshold must be a resolved
massive diagonal branch; the transfer must retain two observable polarization
directions and remain coherent; its leading map must be known or calibrated;
and resolution/background effects must be forward-modelled.  The code tests
the eigenvalue exponents, the exact dark-ray ratio, six-state tomography, and
Weyl bounds for bounded Hermitian perturbations.

4. Adversarial counterexamples and auxiliary observables
---------------------------------------------------------
The audit also constructs cases that invalidate over-broad versions of the
rank statement:

* an incoherent sum of two otherwise rank-one Majorana transfer channels can
  be rank two;
* an off-diagonal unequal-mass channel with a complex Majorana phase can keep
  a transverse S-wave term at threshold;
* finite threshold smearing gives a nonzero small eigenvalue even for a
  single ideal channel;
* mistuning away from the pair-rest double lock increases the projected
  Majorana response.

For completeness, it checks threshold-centered second-harmonic lock-in ratios
(``1/5`` for a beta onset and ``3/7`` for a beta^3 onset) and computes an
illustrative, shape-only Chernoff information.  The latter is an idealized
information benchmark, not a calibrated discovery significance or a
target-specific event forecast.

INDEPENDENT-VALIDATION ROLE
===========================
Every stochastic operation has a fixed seed, and every claimed identity ends
in a numerical assertion with an explicit tolerance.  Passing this script
checks algebra, Lorentz covariance, phase space, threshold powers, tomography,
failure modes, and perturbative robustness using code independent of the
end-to-end synthetic detector pipeline.  It does not replace symbolic proof,
experimental calibration, nuisance profiling, or coverage studies.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np


G = np.diag([1.0, -1.0, -1.0, -1.0])
I4 = np.eye(4, dtype=complex)


def gamma_matrices() -> tuple[list[np.ndarray], np.ndarray]:
    """Construct the Dirac-basis gamma matrices and gamma^5."""

    # Pauli matrices form the off-diagonal spatial blocks in this basis.
    s1 = np.array([[0, 1], [1, 0]], dtype=complex)
    s2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    s3 = np.array([[1, 0], [0, -1]], dtype=complex)
    z = np.zeros((2, 2), dtype=complex)
    eye = np.eye(2, dtype=complex)
    g0 = np.block([[eye, z], [z, -eye]])
    gs = [np.block([[z, s], [-s, z]]) for s in (s1, s2, s3)]
    gammas = [g0, *gs]
    g5 = 1j * gammas[0] @ gammas[1] @ gammas[2] @ gammas[3]
    return gammas, g5


GAMMA, GAMMA5 = gamma_matrices()
PL = (I4 - GAMMA5) / 2.0
PR = (I4 + GAMMA5) / 2.0


def slash(p: np.ndarray) -> np.ndarray:
    """Return gamma^mu p_mu for an upper-index four-vector ``p``."""

    return GAMMA[0] * p[0] - sum(GAMMA[a] * p[a] for a in range(1, 4))


def direct_trace(p1: np.ndarray, p2: np.ndarray, m1: float, m2: float) -> np.ndarray:
    """Evaluate the unintegrated direct spin sum for ``gamma^mu P_L``."""
    out = np.empty((4, 4), dtype=complex)
    for mu in range(4):
        for nu in range(4):
            out[mu, nu] = np.trace(
                (slash(p1) + m1 * I4)
                @ GAMMA[mu]
                @ PL
                @ (slash(p2) - m2 * I4)
                @ GAMMA[nu]
                @ PL
            )
    return out


def lr_cross_trace(p1: np.ndarray, p2: np.ndarray, m1: float, m2: float) -> np.ndarray:
    """Evaluate the crossed trace, predicted to equal ``-2 m1*m2*g``."""
    out = np.empty((4, 4), dtype=complex)
    for mu in range(4):
        for nu in range(4):
            out[mu, nu] = np.trace(
                (slash(p1) + m1 * I4)
                @ GAMMA[mu]
                @ PL
                @ (slash(p2) - m2 * I4)
                @ GAMMA[nu]
                @ PR
            )
    return out


def beta_ij(s: float, m1: float, m2: float) -> float:
    """Return the dimensionless Kallen velocity for two masses at invariant ``s``."""

    x = (1.0 - (m1 + m2) ** 2 / s) * (1.0 - (m1 - m2) ** 2 / s)
    # Guard only against negative roundoff at a physical threshold.
    return float(np.sqrt(max(0.0, x)))


def direct_integrated(q: np.ndarray, m1: float, m2: float) -> np.ndarray:
    """Closed-form phase-space integral of the direct spin tensor."""

    s = float(q @ G @ q)
    beta = beta_ij(s, m1, m2)
    sm = m1 * m1 + m2 * m2
    dm2 = m1 * m1 - m2 * m2
    cg = s * (beta * beta - 3.0 * (1.0 - sm / s))
    cq = 2.0 * (1.0 + sm / s - 2.0 * dm2 * dm2 / (s * s))
    return beta / (24.0 * np.pi) * (cg * G + cq * np.outer(q, q))


def cross_integrated(q: np.ndarray, m1: float, m2: float) -> np.ndarray:
    """Closed-form phase-space integral of the Majorana crossed trace."""

    s = float(q @ G @ q)
    return -beta_ij(s, m1, m2) * m1 * m2 / (4.0 * np.pi) * G


def equal_mass_tensor(q: np.ndarray, m: float, majorana: bool) -> np.ndarray:
    """Return the matched Dirac or diagonal equal-mass Majorana tensor."""

    d = direct_integrated(q, m, m)
    return d - cross_integrated(q, m, m) if majorana else d


def majorana_general_tensor(q: np.ndarray, m1: float, m2: float, a: complex) -> np.ndarray:
    """Return an unequal-mass Majorana tensor in the ordered-pair convention."""

    # The exchange term is phase-sensitive through Re(a^2), not merely |a|^2.
    return abs(a) ** 2 * direct_integrated(q, m1, m2) - np.real(a * a) * cross_integrated(q, m1, m2)


def boost_z(p: np.ndarray, rapidity: float) -> np.ndarray:
    """Apply an active Lorentz boost along +z using rapidity coordinates."""

    c, s = np.cosh(rapidity), np.sinh(rapidity)
    return np.array([c * p[0] + s * p[3], p[1], p[2], s * p[0] + c * p[3]])


def phase_space_mc(s: float, m1: float, m2: float, rapidity: float, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Numerically integrate direct and crossed traces over exact two-body phase space."""

    rng = np.random.default_rng(seed)
    root_s = np.sqrt(s)
    e1 = (s + m1 * m1 - m2 * m2) / (2.0 * root_s)
    e2 = (s + m2 * m2 - m1 * m1) / (2.0 * root_s)
    k = root_s * beta_ij(s, m1, m2) / 2.0
    acc_d = np.zeros((4, 4), dtype=complex)
    acc_x = np.zeros((4, 4), dtype=complex)
    # Antithetic directions cancel the integrated epsilon tensor exactly pairwise.
    for _ in range(n // 2):
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        for sign in (1.0, -1.0):
            vv = sign * v
            p1 = boost_z(np.r_[e1, k * vv], rapidity)
            p2 = boost_z(np.r_[e2, -k * vv], rapidity)
            acc_d += direct_trace(p1, p2, m1, m2)
            acc_x += lr_cross_trace(p1, p2, m1, m2)
    phi2 = beta_ij(s, m1, m2) / (8.0 * np.pi)
    return phi2 * acc_d / n, phi2 * acc_x / n


def response(j_upper: np.ndarray, tensor_upper: np.ndarray) -> np.ndarray:
    """Contract a covariant neutrino tensor into a two-output Hermitian response."""

    # J is stored with an upper Lorentz index; multiplication by G lowers it.
    j_lower = j_upper @ G
    r = j_lower @ tensor_upper @ j_lower.conj().T
    # Symmetrization removes only floating-point anti-Hermitian residue.
    return (r + r.conj().T) / 2.0


def ordered_eigs(r: np.ndarray) -> np.ndarray:
    """Return Hermitian eigenvalues from largest to smallest."""

    return np.linalg.eigvalsh(r)[::-1]


def log_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Fit the power-law exponent in ``y proportional to x**slope``."""

    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def dark_vector(c: np.ndarray) -> np.ndarray:
    """Construct the normalized two-component analyzer ray orthogonal to ``c``."""

    e = np.array([-np.conj(c[1]), np.conj(c[0])])
    return e / np.linalg.norm(e)


def tomography_rates(r: np.ndarray) -> dict[str, float]:
    """Evaluate six canonical polarization projections of a 2x2 response."""

    states = {
        "H": np.array([1, 0], complex),
        "V": np.array([0, 1], complex),
        "D": np.array([1, 1], complex) / np.sqrt(2),
        "A": np.array([1, -1], complex) / np.sqrt(2),
        "R": np.array([1, -1j], complex) / np.sqrt(2),
        "L": np.array([1, 1j], complex) / np.sqrt(2),
    }
    return {k: float(np.real(v.conj() @ r @ v)) for k, v in states.items()}


def reconstruct_from_six(x: dict[str, float]) -> np.ndarray:
    """Reconstruct a Hermitian response from H/V, D/A, and R/L projections."""

    # With the state convention above, R-L = 2 Im(R_HV).
    off = (x["D"] - x["A"] + 1j * (x["R"] - x["L"])) / 2.0
    return np.array([[x["H"], off], [np.conj(off), x["V"]]], complex)


@dataclass
class Audit:
    """Serializable collection of all independently asserted audit diagnostics."""

    clifford_max_error: float
    cross_trace_max_error: float
    phase_space_direct_rel_error: float
    phase_space_cross_rel_error: float
    dirac_small_eigenvalue_slope: float
    majorana_small_eigenvalue_slope: float
    dark_projector_ratio_rel_error: float
    tomography_rel_error: float
    majorana_single_channel_rank_ratio: float
    majorana_two_channel_rank_ratio: float
    offdiagonal_real_phase_rank_ratio: float
    offdiagonal_complex_phase_rank_ratio: float
    double_lock_ratio_at_beta_015: float
    double_lock_mistuning_enhancement: float
    smeared_majorana_rank_ratio: float
    harmonic_ratio_dirac: float
    harmonic_ratio_majorana_projected: float
    ideal_shape_chernoff_information: float
    ideal_shape_events_for_5sigma_bound: float
    weyl_dirac_lower_bound_holds: bool
    weyl_majorana_upper_bound_holds: bool


def run() -> Audit:
    """Execute the complete deterministic audit and return its diagnostics."""

    rng = np.random.default_rng(20260810)

    # 1. Representation sanity check: {gamma^mu,gamma^nu}=2g^{mu nu}I.
    clifford = 0.0
    for mu in range(4):
        for nu in range(4):
            target = 2.0 * G[mu, nu] * I4
            clifford = max(clifford, float(np.max(np.abs(GAMMA[mu] @ GAMMA[nu] + GAMMA[nu] @ GAMMA[mu] - target))))

    # 2. Test the crossed trace at a generic unequal-mass boosted point.
    m1, m2, s, rapidity = 0.011, 0.027, 0.009, 0.63
    root_s = np.sqrt(s)
    e1 = (s + m1 * m1 - m2 * m2) / (2 * root_s)
    e2 = (s + m2 * m2 - m1 * m1) / (2 * root_s)
    k = root_s * beta_ij(s, m1, m2) / 2
    direction = np.array([0.2, -0.7, 0.6855654600401044])
    p1 = boost_z(np.r_[e1, k * direction], rapidity)
    p2 = boost_z(np.r_[e2, -k * direction], rapidity)
    x_num = lr_cross_trace(p1, p2, m1, m2)
    x_exact = -2.0 * m1 * m2 * G
    cross_trace_err = float(np.max(np.abs(x_num - x_exact)))

    # 3. Compare angular quadrature of raw traces with the covariant integral.
    q = boost_z(np.array([root_s, 0.0, 0.0, 0.0]), rapidity)
    mc_d, mc_x = phase_space_mc(s, m1, m2, rapidity, n=40000, seed=91452)
    an_d, an_x = direct_integrated(q, m1, m2), cross_integrated(q, m1, m2)
    rel_d = float(np.linalg.norm(mc_d - an_d) / np.linalg.norm(an_d))
    rel_x = float(np.linalg.norm(mc_x - an_x) / np.linalg.norm(an_x))

    # 4. A random but reproducible transfer tests the theorem without choosing
    # a privileged polarization basis or special atomic-current orientation.
    j = rng.normal(size=(2, 4)) + 1j * rng.normal(size=(2, 4))
    m = 0.03
    st = 4.0 * m * m
    rel_delta = np.geomspace(1e-7, 3e-3, 20)
    d_small, m_small = [], []
    for rd in rel_delta:
        ss = st * (1.0 + rd)
        qq = boost_z(np.array([np.sqrt(ss), 0.0, 0.0, 0.0]), 0.47)
        d_small.append(ordered_eigs(response(j, equal_mass_tensor(qq, m, False)))[1])
        m_small.append(ordered_eigs(response(j, equal_mass_tensor(qq, m, True)))[1])
    slope_d = log_slope(rel_delta, np.array(d_small))
    slope_m = log_slope(rel_delta, np.array(m_small))

    # 5. Project out c=J_mu q^mu and verify the exact dark-ray M/D ratio.
    beta_target = 0.23
    ss = st / (1.0 - beta_target * beta_target)
    qq = boost_z(np.array([np.sqrt(ss), 0.0, 0.0, 0.0]), 0.47)
    c = (j @ G) @ qq
    e = dark_vector(c)
    rd = float(np.real(e.conj() @ response(j, equal_mass_tensor(qq, m, False)) @ e))
    rm = float(np.real(e.conj() @ response(j, equal_mass_tensor(qq, m, True)) @ e))
    ratio_exact = (ss - 4.0 * m * m) / (ss - m * m)
    ratio_err = abs(rm / rd - ratio_exact) / ratio_exact

    # 6. Verify that the six physical analyzer settings are tomographically complete.
    r_random = j @ j.conj().T
    rates = tomography_rates(r_random)
    r_rec = reconstruct_from_six(rates)
    tomo_err = float(np.linalg.norm(r_rec - r_random) / np.linalg.norm(r_random))

    # 7. At threshold divide out beta and use a tiny positive offset to avoid
    # the exactly zero phase-space point.  One coherent Majorana channel is
    # rank one at leading order; an incoherent sum need not be.
    ss0 = st * (1 + 1e-10)
    q0 = boost_z(np.array([np.sqrt(ss0), 0.0, 0.0, 0.0]), 0.47)
    beta0 = beta_ij(ss0, m, m)
    cm_one = response(j, equal_mass_tensor(q0, m, True)) / beta0
    single_ratio = float(ordered_eigs(cm_one)[1] / ordered_eigs(cm_one)[0])
    j2 = rng.normal(size=(2, 4)) + 1j * rng.normal(size=(2, 4))
    cm_two = cm_one + response(j2, equal_mass_tensor(q0, m, True)) / beta0
    two_ratio = float(ordered_eigs(cm_two)[1] / ordered_eigs(cm_two)[0])

    # 8. Adversarial unequal-mass test: cancellation occurs for a real weak
    # coefficient, while a complex phase leaves a transverse S-wave.
    mo1, mo2 = 0.018, 0.041
    sto = (mo1 + mo2) ** 2
    sso = sto * (1 + 1e-10)
    qo = np.array([np.sqrt(sso), 0.0, 0.0, 0.0])
    bo = beta_ij(sso, mo1, mo2)
    co_real = response(j, majorana_general_tensor(qo, mo1, mo2, 1.0 + 0j)) / bo
    co_phase = response(j, majorana_general_tensor(qo, mo1, mo2, np.exp(0.25j * np.pi))) / bo
    off_real = float(ordered_eigs(co_real)[1] / ordered_eigs(co_real)[0])
    off_phase = float(ordered_eigs(co_phase)[1] / ordered_eigs(co_phase)[0])

    # 9. Pair-rest double lock with a pure spatial current.  At q_vec=0 the
    # beta-vs-beta^3 ratio applies to the full response, not only a dark ray.
    j_spatial = j.copy()
    j_spatial[:, 0] = 0.0
    beta_lock = 0.15
    ssl = st / (1 - beta_lock**2)
    ql = np.array([np.sqrt(ssl), 0.0, 0.0, 0.0])
    trd_lock = float(np.trace(response(j_spatial, equal_mass_tensor(ql, m, False))).real)
    trm_lock = float(np.trace(response(j_spatial, equal_mass_tensor(ql, m, True))).real)
    lock_ratio = trm_lock / trd_lock
    delta_lock = ssl - st
    qz = np.sqrt(delta_lock)
    q_mistuned = np.array([np.sqrt(ssl + qz * qz), 0.0, 0.0, qz])
    trm_mistuned = float(np.trace(response(j_spatial, equal_mass_tensor(q_mistuned, m, True))).real)
    mistuning_enhancement = trm_mistuned / trm_lock

    # 10. A known resolution kernel must be forward-modelled.  Even a single
    # ideal channel acquires a finite small eigenvalue after threshold smearing.
    sigma_rel = 1e-4
    grid = np.linspace(1e-10, 6 * sigma_rel, 500)
    weights = np.exp(-0.5 * (grid / sigma_rel) ** 2)
    weights /= np.trapezoid(weights, grid)
    r_smear = np.zeros((2, 2), complex)
    for aa, ww in zip(grid, weights):
        ssi = st * (1 + aa)
        qi = boost_z(np.array([np.sqrt(ssi), 0.0, 0.0, 0.0]), 0.47)
        r_smear += ww * response(j, equal_mass_tensor(qi, m, True))
    r_smear *= grid[1] - grid[0]
    smear_ratio = float(ordered_eigs(r_smear)[1] / ordered_eigs(r_smear)[0])

    # 11. Threshold-centered lock-in modulation
    # f(phi)=Theta(cos(phi))*cos(phi)^p distinguishes p=1/2 from p=3/2.
    phi = np.linspace(-np.pi, np.pi, 400001)
    cp = np.clip(np.cos(phi), 0.0, None)
    harmonic = []
    for power in (0.5, 1.5):
        f = cp**power
        harmonic.append(float(np.trapezoid(f * np.cos(2 * phi), phi) / np.trapezoid(f, phi)))

    # 12. Illustrative shape-only Chernoff information after normalizing away
    # the total rate.  This intentionally omits experimental nuisance profiling.
    rd_mat = response(j, equal_mass_tensor(qq, m, False))
    rm_mat = response(j, equal_mass_tensor(qq, m, True))
    vd = np.array(list(tomography_rates(rd_mat).values()))
    vm = np.array(list(tomography_rates(rm_mat).values()))
    pd, pm = vd / vd.sum(), vm / vm.sum()
    ts = np.linspace(0.0, 1.0, 20001)
    overlaps = np.array([np.sum(pd**tt * pm ** (1.0 - tt)) for tt in ts])
    chernoff = float(-np.log(overlaps.min()))
    p5 = 2.87e-7
    n5 = float(np.log(0.5 / p5) / chernoff)

    # 13. Numerical Weyl audit: operator-norm-bounded Hermitian perturbations
    # cannot move the ordered eigenvalues by more than their norm.
    cd = response(j, equal_mass_tensor(q0, m, False)) / beta0
    cm = cm_one
    ed_raw = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    ed_raw = (ed_raw + ed_raw.conj().T) / 2
    em_raw = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    em_raw = (em_raw + em_raw.conj().T) / 2
    eps_d = 0.07 * ordered_eigs(cd)[1]
    eps_m = 0.05 * ordered_eigs(cd)[1]
    ed = eps_d * ed_raw / np.linalg.norm(ed_raw, 2)
    em = eps_m * em_raw / np.linalg.norm(em_raw, 2)
    weyl_d = ordered_eigs(cd + ed)[1] >= ordered_eigs(cd)[1] - eps_d - 1e-12
    weyl_m = ordered_eigs(cm + em)[1] <= eps_m + abs(ordered_eigs(cm)[1]) + 1e-12

    # Assemble a machine-readable record before applying the fail-closed checks.
    audit = Audit(
        clifford_max_error=clifford,
        cross_trace_max_error=cross_trace_err,
        phase_space_direct_rel_error=rel_d,
        phase_space_cross_rel_error=rel_x,
        dirac_small_eigenvalue_slope=slope_d,
        majorana_small_eigenvalue_slope=slope_m,
        dark_projector_ratio_rel_error=float(ratio_err),
        tomography_rel_error=tomo_err,
        majorana_single_channel_rank_ratio=single_ratio,
        majorana_two_channel_rank_ratio=two_ratio,
        offdiagonal_real_phase_rank_ratio=off_real,
        offdiagonal_complex_phase_rank_ratio=off_phase,
        double_lock_ratio_at_beta_015=float(lock_ratio),
        double_lock_mistuning_enhancement=float(mistuning_enhancement),
        smeared_majorana_rank_ratio=smear_ratio,
        harmonic_ratio_dirac=harmonic[0],
        harmonic_ratio_majorana_projected=harmonic[1],
        ideal_shape_chernoff_information=chernoff,
        ideal_shape_events_for_5sigma_bound=n5,
        weyl_dirac_lower_bound_holds=bool(weyl_d),
        weyl_majorana_upper_bound_holds=bool(weyl_m),
    )

    # These assertions are the executable acceptance criteria.  Their margins
    # cover Monte Carlo/quadrature error but are tight enough to catch sign,
    # normalization, index-position, threshold-power, and rank regressions.
    assert audit.clifford_max_error < 1e-13
    assert audit.cross_trace_max_error < 1e-12
    assert audit.phase_space_direct_rel_error < 1.5e-2
    assert audit.phase_space_cross_rel_error < 1e-12
    assert abs(audit.dirac_small_eigenvalue_slope - 0.5) < 0.03
    assert abs(audit.majorana_small_eigenvalue_slope - 1.5) < 0.04
    assert audit.dark_projector_ratio_rel_error < 1e-10
    assert audit.tomography_rel_error < 1e-13
    assert audit.majorana_single_channel_rank_ratio < 1e-8
    assert audit.majorana_two_channel_rank_ratio > 1e-3
    assert audit.offdiagonal_real_phase_rank_ratio < 1e-8
    assert audit.offdiagonal_complex_phase_rank_ratio > 1e-3
    assert abs(audit.double_lock_ratio_at_beta_015 - 4 * beta_lock**2 / (3 + beta_lock**2)) < 1e-12
    assert audit.double_lock_mistuning_enhancement > 1.0
    assert audit.smeared_majorana_rank_ratio > 0.0
    assert abs(audit.harmonic_ratio_dirac - 0.2) < 2e-5
    assert abs(audit.harmonic_ratio_majorana_projected - 3 / 7) < 2e-5
    assert audit.ideal_shape_chernoff_information > 0.0
    assert audit.weyl_dirac_lower_bound_holds
    assert audit.weyl_majorana_upper_bound_holds
    return audit


if __name__ == "__main__":
    print(json.dumps(asdict(run()), indent=2, sort_keys=True))
