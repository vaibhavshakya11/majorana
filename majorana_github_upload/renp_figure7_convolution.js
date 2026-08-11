#!/usr/bin/env node
/*
 * RENP FIGURE 7: GAUSSIAN SMEARING AND LEADING-POWER LEAKAGE
 * ============================================================================
 *
 * This small, dependency-free program reproduces the solid coordinate tables in
 * Fig. 7 of the manuscript.  It is separate from the much larger neutrino-tensor
 * stress suite so that the plotted model can be checked without Python, NumPy, or
 * SciPy.
 *
 * MATHEMATICAL MODEL
 * ------------------
 * Let x = delta/m^2, where delta = s - 4m^2.  The ideal threshold templates are
 *
 *     f_D(x) = x_+^(1/2),       f_M(x) = x_+^(3/2),
 *     x_+ = max(x, 0).
 *
 * A Gaussian setting error of width sigma_x gives
 *
 *   fbar_alpha(x;sigma_x)
 *     = integral_0^infinity dy / (sqrt(2 pi) sigma_x)
 *       exp[-(x-y)^2/(2 sigma_x^2)] y^alpha.
 *
 * The displayed stress model then uses
 *
 *     f_D^obs = fbar_(1/2),
 *     f_M^obs = fbar_(3/2) + epsilon_leak fbar_(1/2).
 *
 * The added term is a deliberately injected rank-filling component with the
 * leading Dirac threshold power.  It is not a target prediction.  Likewise, the
 * one-dimensional Gaussian is a transparent exponent test, not a replacement for
 * a measured joint momentum/optical resolution kernel.
 *
 * NUMERICS
 * --------
 * With u=(y-x)/sigma_x, the Gaussian is centered at u=0 for every x.  We integrate
 * from u=-x/sigma_x (the physical y=0 boundary) to u=12; the omitted normal tail is
 * far below the printed precision.  Adaptive Simpson integration is performed on
 * the two intervals separated at u=0 so that the narrow Gaussian peak is always
 * sampled.  The integrand is nonnegative, and the y=0 endpoint is evaluated as 0.
 *
 * Running this file prints LaTeX/PGFPlots coordinate blocks for direct comparison
 * with sections_7_10.tex.  No files are written.
 */

'use strict';

const SIGMA_X = 0.003;
const EPSILON_LEAK = 0.025;
const SQRT_TWO_PI = Math.sqrt(2 * Math.PI);

// The exact x grid printed in Fig. 7, including the beta=0.15 benchmark x=0.09207.
const X_VALUES = [
  0.00015, 0.0003, 0.00075, 0.0015, 0.003, 0.006, 0.015,
  0.03, 0.06, 0.09207, 0.2, 0.4, 1.0,
];

function simpson(f, a, b, fa, fm, fb) {
  // Simpson's rule on one interval when endpoint and midpoint values are known.
  return ((b - a) / 6) * (fa + 4 * fm + fb);
}

function adaptiveSimpson(f, a, b, tolerance = 2e-13, maxDepth = 28) {
  // Recursive error control: the difference between one panel and two half-panels
  // estimates fifteen times the Simpson truncation error.
  if (!(b > a)) return 0;

  const midpoint = (a + b) / 2;
  const fa = f(a);
  const fm = f(midpoint);
  const fb = f(b);
  const whole = simpson(f, a, b, fa, fm, fb);

  function refine(left, right, fLeft, fMiddle, fRight, coarse, tol, depth) {
    const middle = (left + right) / 2;
    const leftMiddle = (left + middle) / 2;
    const rightMiddle = (middle + right) / 2;
    const fLeftMiddle = f(leftMiddle);
    const fRightMiddle = f(rightMiddle);
    const leftPanel = simpson(f, left, middle, fLeft, fLeftMiddle, fMiddle);
    const rightPanel = simpson(f, middle, right, fMiddle, fRightMiddle, fRight);
    const fine = leftPanel + rightPanel;
    const correction = fine - coarse;

    if (depth <= 0 || Math.abs(correction) <= 15 * tol) {
      return fine + correction / 15;
    }

    return refine(
      left, middle, fLeft, fLeftMiddle, fMiddle,
      leftPanel, tol / 2, depth - 1,
    ) + refine(
      middle, right, fMiddle, fRightMiddle, fRight,
      rightPanel, tol / 2, depth - 1,
    );
  }

  return refine(a, b, fa, fm, fb, whole, tolerance, maxDepth);
}

function gaussianConvolution(x, alpha, sigma = SIGMA_X) {
  // After y=x+sigma*u, the 1/sigma Jacobian cancels and only the standard-normal
  // density remains.  max(y,0) protects the boundary against roundoff.
  const lower = -x / sigma;
  const upper = 12;
  const integrand = (u) => {
    const y = x + sigma * u;
    if (y <= 0) return 0;
    return Math.exp(-0.5 * u * u) * Math.pow(y, alpha) / SQRT_TWO_PI;
  };

  // Splitting at the Gaussian center prevents an adaptive method from overlooking
  // the peak when x/sigma is large.
  if (lower < 0) {
    return adaptiveSimpson(integrand, lower, 0)
      + adaptiveSimpson(integrand, 0, upper);
  }
  return adaptiveSimpson(integrand, lower, upper);
}

function printCoordinateBlock(label, values) {
  console.log(`${label} coordinates {`);
  for (const { x, y } of values) {
    console.log(`  (${x.toPrecision(8)},${y.toPrecision(12)})`);
  }
  console.log('};');
}

const dirac = [];
const majorana = [];

for (const x of X_VALUES) {
  const fD = gaussianConvolution(x, 0.5);
  const fM = gaussianConvolution(x, 1.5) + EPSILON_LEAK * fD;
  dirac.push({ x, y: fD });
  majorana.push({ x, y: fM });
}

console.log(`# sigma_x=${SIGMA_X}, epsilon_leak=${EPSILON_LEAK}`);
printCoordinateBlock('Dirac Gaussian-convolved', dirac);
printCoordinateBlock('Majorana convolved + leakage', majorana);

