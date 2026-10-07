//! Dependency-free quantile functions for the percentile-growth model.
//!
//! The Python reference computes `scipy.stats.t.ppf(p, 9_999_999)`. At that
//! many degrees of freedom the Student-t quantile differs from the standard
//! normal quantile by about 6.6e-8 relative, which is far above the 1e-9
//! golden tolerance, so the t correction is applied rather than dropped.
//!
//! Method: solve `Phi(z) = p` by Newton's method, with `Phi` built from an
//! `erfc` that uses a convergent series for small arguments and a continued
//! fraction for large ones (no coefficient tables), then apply the
//! Cornish-Fisher expansion of the t quantile in `1/df`.

use std::f64::consts::{PI, SQRT_2};

/// `erf(x)` for `0 <= x < 2` by the non-cancelling series
/// `(2/sqrt(pi)) e^{-x^2} sum_n 2^n x^{2n+1} / (2n+1)!!`.
fn erf_series(x: f64) -> f64 {
    let two_x2 = 2.0 * x * x;
    let mut term = x;
    let mut sum = x;
    let mut n = 0.0;
    while term > 1e-18 * sum {
        n += 1.0;
        term *= two_x2 / (2.0 * n + 1.0);
        sum += term;
    }
    2.0 / PI.sqrt() * (-x * x).exp() * sum
}

/// `erfc(x)` for `x >= 2` by the continued fraction
/// `e^{-x^2}/sqrt(pi) / (x + (1/2)/(x + (2/2)/(x + (3/2)/(x + ...))))`,
/// evaluated backwards to a fixed depth.
fn erfc_cf(x: f64) -> f64 {
    let mut f = x;
    for k in (1..=300).rev() {
        f = x + (k as f64 / 2.0) / f;
    }
    (-x * x).exp() / (PI.sqrt() * f)
}

/// `erfc(x)` for `x >= 0`.
fn erfc_nonneg(x: f64) -> f64 {
    if x < 2.0 {
        1.0 - erf_series(x)
    } else {
        erfc_cf(x)
    }
}

/// Standard normal CDF for `z <= 0` (the lower tail used by the solver).
fn phi_lower(z: f64) -> f64 {
    0.5 * erfc_nonneg(-z / SQRT_2)
}

/// Standard normal density.
fn pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * PI).sqrt()
}

/// Standard normal quantile. `p <= 0` gives `-inf`, `p >= 1` gives `+inf`,
/// NaN gives NaN, and `0.5` gives exactly `0.0`.
pub fn normal_quantile(p: f64) -> f64 {
    if p.is_nan() {
        return f64::NAN;
    }
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p == 0.5 {
        return 0.0;
    }
    // Solve in the lower tail. `1 - p` is exact for p in [0.5, 1].
    if p > 0.5 {
        return -normal_quantile(1.0 - p);
    }
    let mut z = -(-2.0 * p.ln()).sqrt();
    for _ in 0..100 {
        let step = (phi_lower(z) - p) / pdf(z);
        z -= step;
        if step.abs() <= 1e-16 * z.abs().max(1.0) {
            break;
        }
    }
    z
}

/// Student-t quantile for large degrees of freedom: the normal quantile plus
/// the first two terms of the Cornish-Fisher expansion in `1/df`
/// (Abramowitz & Stegun 26.7.5). Intended for `df` around 1e7, where the
/// neglected terms are far below double precision.
pub fn t_quantile_large_df(p: f64, df: f64) -> f64 {
    let z = normal_quantile(p);
    if !z.is_finite() {
        return z;
    }
    let z2 = z * z;
    let g1 = (z2 + 1.0) * z / 4.0;
    let g2 = ((5.0 * z2 + 16.0) * z2 + 3.0) * z / 96.0;
    z + g1 / df + g2 / (df * df)
}

#[cfg(test)]
mod tests {
    use super::*;

    // scipy.stats.t.ppf(p, 9999999), generated with:
    //   uv run --no-sync python -c "from scipy.stats import t; ..."
    // scipy's own value at p and 1-p differ in the last ~3 digits at the
    // far tail, so the comparison tolerance is 1e-11 relative.
    const SCIPY_T: &[(f64, f64)] = &[
        (1e-05, -4.264892839929923),
        (0.001, -3.090233121180993),
        (0.05, -1.6448537793284168),
        (0.1, -1.2815516502030913),
        (0.25, -0.6744897747295779),
        (0.75, 0.6744897747295779),
        (0.9, 1.2815516502030913),
        (0.95, 1.6448537793284164),
        (0.99, 2.3263482469483594),
        (0.999, 3.090233121180993),
        (0.99999, 4.264892839930939),
    ];

    #[test]
    fn t_quantile_matches_scipy_at_df_9999999() {
        for &(p, expected) in SCIPY_T {
            let got = t_quantile_large_df(p, 9_999_999.0);
            let rel = ((got - expected) / expected).abs();
            assert!(rel <= 1e-11, "p={p}: expected {expected}, got {got} (rel {rel:e})");
        }
    }

    #[test]
    fn median_is_exactly_zero() {
        assert_eq!(t_quantile_large_df(0.5, 9_999_999.0), 0.0);
        assert_eq!(normal_quantile(0.5), 0.0);
    }

    #[test]
    fn normal_quantile_is_odd_symmetric() {
        // dyadic p so that 1 - p is exact in floating point
        for p in [2f64.powi(-17), 2f64.powi(-10), 0.0625, 0.125, 0.25, 0.375] {
            let (lo, hi) = (normal_quantile(p), normal_quantile(1.0 - p));
            assert!((lo + hi).abs() <= 1e-14 * lo.abs().max(1.0), "p={p}: {lo} vs {hi}");
        }
    }

    #[test]
    fn normal_quantile_known_values() {
        // Standard references: z(0.975) = 1.959963984540054, z(0.995) = 2.5758293035489004
        assert!((normal_quantile(0.975) - 1.959963984540054).abs() <= 1e-13);
        assert!((normal_quantile(0.995) - 2.5758293035489004).abs() <= 1e-13);
    }

    #[test]
    fn out_of_range_and_nan() {
        assert_eq!(normal_quantile(0.0), f64::NEG_INFINITY);
        assert_eq!(normal_quantile(1.0), f64::INFINITY);
        assert!(normal_quantile(f64::NAN).is_nan());
        assert!(t_quantile_large_df(f64::NAN, 9_999_999.0).is_nan());
    }
}
