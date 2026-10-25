//! Foliar moisture content and effect, one function per quantity
//! (`python/equations/foliar_moisture.py`), and the calendar helpers they need.
//!
//! The quantities chain as latn -> d0, dj -> nd -> fmc -> fme;
//! [`calc_foliar_moisture`] runs the chain.

/// Days in each month of a non-leap year.
const MONTH_DAYS: [i64; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

/// Days elapsed before the first of each month in a non-leap year.
const CUM_DAYS: [i64; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];

// Field names mirror the Python `FoliarMoistureResult` fields (`fmc` included).
#[allow(clippy::struct_field_names)]
pub(crate) struct FoliarMoisture {
    pub latn: f64,
    pub d0: f64,
    pub dj: f64,
    pub nd: f64,
    pub fmc: f64,
    pub fme: f64,
}

/// Day of year from YYYYMMDD — `datetime.strptime(...).timetuple().tm_yday`.
/// NaN for an invalid date.
// `mth` is 1..=12 after the validity check; `doy` is at most 366, exact in f64.
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]
fn day_of_year(wx_date: i64) -> f64 {
    if !is_valid_wx_date(wx_date) {
        return f64::NAN;
    }
    let y = wx_date / 10_000;
    let mth = (wx_date / 100 % 100) as usize;
    let d = wx_date % 100;
    let mut doy = CUM_DAYS[mth - 1] + d;
    if mth > 2 && is_leap_year(y) {
        doy += 1;
    }
    doy as f64
}

/// An override value as the calculation uses it: a negative value (documented as -1) asks for the
/// derived value, so it is no override; NaN is a missing value and stays one.
#[inline]
fn given(override_value: Option<f64>) -> Option<f64> {
    override_value.filter(|v| v.is_nan() || *v >= 0.0)
}

/// `foliar_moisture.calc_d0`: the date of minimum foliar moisture content,
/// rounded to mimic the cffdrs R package (numpy round = ties to even). NaN
/// where the latitude or the normalized latitude is missing. Not limited to a
/// calendar day.
#[inline]
pub(crate) fn calc_d0(lat: f64, latn: f64, elevation: f64) -> f64 {
    if lat.is_nan() || latn.is_nan() {
        return f64::NAN;
    }
    if elevation > 0.0 {
        142.1 * (lat / latn) + 0.0172 * elevation
    } else {
        151.0 * (lat / latn)
    }
    .round_ties_even()
}

/// `foliar_moisture.calc_dj`: the day of the year of the modelled day, NaN
/// for an invalid date and 0 where the normalized latitude is not finite.
#[inline]
pub(crate) fn calc_dj(latn: f64, wx_date: i64) -> f64 {
    if latn.is_nan() {
        f64::NAN
    } else if latn.is_finite() {
        day_of_year(wx_date)
    } else {
        0.0
    }
}

/// `foliar_moisture.calc_fmc`: the foliar moisture content (%) from `nd`
/// (NaN in, NaN out).
#[inline]
pub(crate) fn calc_fmc(nd: f64) -> f64 {
    if nd.is_nan() {
        f64::NAN
    } else if nd < 30.0 {
        85.0 + 0.0189 * nd * nd
    } else if nd < 50.0 {
        32.9 + 3.17 * nd - 0.0288 * nd * nd
    } else {
        120.0
    }
}

/// `foliar_moisture.calc_fme`: the foliar moisture effect from the foliar
/// moisture content (NaN in, NaN out).
#[inline]
pub(crate) fn calc_fme(fmc: f64) -> f64 {
    1000.0 * (1.5 - 0.00275 * fmc).powi(4) / (460.0 + 25.9 * fmc)
}

/// `foliar_moisture.calc_foliar_moisture`: the chain of the functions above.
/// An explicit `d0_override`, `dj_override` or `fmc_override` replaces the
/// derived value; the fme follows the fmc that is used. `Some(NaN)` is a
/// missing value and a negative value (documented as -1) asks for the
/// derived value, as `None` does.
pub(crate) fn calc_foliar_moisture(
    lat: f64,
    abs_long: f64,
    elevation: f64,
    wx_date: i64,
    d0_override: Option<f64>,
    dj_override: Option<f64>,
    fmc_override: Option<f64>,
) -> FoliarMoisture {
    let latn = calc_latn(elevation, abs_long);
    let d0 = given(d0_override).unwrap_or_else(|| calc_d0(lat, latn, elevation));
    let dj = given(dj_override).unwrap_or_else(|| calc_dj(latn, wx_date));
    let nd = calc_nd(dj, d0);
    let fmc = given(fmc_override).unwrap_or_else(|| calc_fmc(nd));
    let fme = calc_fme(fmc);
    FoliarMoisture {
        latn,
        d0,
        dj,
        nd,
        fmc,
        fme,
    }
}

/// `foliar_moisture.calc_latn`: the normalized latitude. Missing geography is
/// a masked cell in Python and stays NaN here: a NaN elevation or longitude
/// masks it.
#[inline]
pub(crate) fn calc_latn(elevation: f64, abs_long: f64) -> f64 {
    if elevation.is_nan() || abs_long.is_nan() {
        f64::NAN
    } else if elevation > 0.0 {
        43.0 + 33.7 * (-0.0351 * (150.0 - abs_long)).exp()
    } else {
        46.0 + 23.4 * (-0.036 * (150.0 - abs_long)).exp()
    }
}

/// `foliar_moisture.calc_nd`: the number of days between the modelled day and
/// the date of minimum foliar moisture.
#[inline]
pub(crate) fn calc_nd(dj: f64, d0: f64) -> f64 {
    (dj - d0).abs()
}

/// Gregorian leap year: divisible by 4, except centuries not divisible by 400.
pub(crate) fn is_leap_year(y: i64) -> bool {
    y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)
}

/// Whether `wx_date` is a real `YYYYMMDD` calendar date (year 1-9999), as
/// `datetime.strptime(str(wx_date), '%Y%m%d')` requires. Callers should check
/// this once before a grid pass; the core itself treats an invalid date as
/// missing (NaN) rather than panicking.
// `mth` is range-checked to 1..=12 just above, so the index cast cannot go negative.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
#[must_use]
pub fn is_valid_wx_date(wx_date: i64) -> bool {
    if !(10_000_101..=99_991_231).contains(&wx_date) {
        return false;
    }
    let y = wx_date / 10_000;
    let mth = wx_date / 100 % 100;
    let d = wx_date % 100;
    if !(1..=12).contains(&mth) {
        return false;
    }
    let days = MONTH_DAYS[(mth - 1) as usize] + i64::from(mth == 2 && is_leap_year(y));
    (1..=days).contains(&d)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- fmc tests ----
    #[test]
    fn fmc_follows_the_three_branches_of_nd() {
        assert_eq!(calc_fmc(0.0), 85.0);
        assert_eq!(calc_fmc(10.0), 85.0 + 0.0189 * 100.0);
        assert_eq!(calc_fmc(30.0), 32.9 + 3.17 * 30.0 - 0.0288 * 900.0);
        assert_eq!(calc_fmc(50.0), 120.0);
        assert_eq!(calc_fmc(300.0), 120.0);
        assert!(calc_fmc(f64::NAN).is_nan());
    }

    #[test]
    fn an_fmc_override_replaces_the_derived_fmc_and_fme_follows_it() {
        let r = calc_foliar_moisture(55.0, 110.0, 500.0, 20_230_615, None, None, Some(100.0));
        assert_eq!((r.d0, r.dj, r.nd), (161.0, 166.0, 5.0));
        assert_eq!(r.fmc, 100.0);
        assert_eq!(r.fme, calc_fme(100.0));
        let missing =
            calc_foliar_moisture(55.0, 110.0, 500.0, 20_230_615, None, None, Some(f64::NAN));
        assert!(missing.fmc.is_nan() && missing.fme.is_nan());
        assert_eq!(missing.nd, 5.0);
    }

    // ---- other tests ----
    #[test]
    fn march_first_counts_the_leap_day() {
        assert_eq!(day_of_year(20_240_301), 61.0); // 2024 is a leap year
        assert_eq!(day_of_year(20_230_301), 60.0);
        assert_eq!(day_of_year(19_000_301), 60.0); // century, not leap
        assert_eq!(day_of_year(20_000_301), 61.0); // 400-year century, leap
    }

    #[test]
    fn december_31_is_the_last_day() {
        assert_eq!(day_of_year(20_241_231), 366.0);
        assert_eq!(day_of_year(20_231_231), 365.0);
    }

    #[test]
    fn extreme_dates_are_invalid_without_panicking() {
        assert!(!is_valid_wx_date(i64::MIN));
        assert!(!is_valid_wx_date(i64::MAX));
        assert!(day_of_year(i64::MIN).is_nan());
        assert!(day_of_year(i64::MAX).is_nan());
    }

    #[test]
    fn february_29_only_in_leap_years() {
        assert!(is_valid_wx_date(20_240_229));
        assert!(!is_valid_wx_date(20_230_229));
        assert!(!is_valid_wx_date(19_000_229));
        assert_eq!(day_of_year(20_240_229), 60.0);
    }

    #[test]
    fn latn_uses_the_elevation_branch_and_follows_a_missing_longitude() {
        let above = 43.0 + 33.7 * (-0.0351 * (150.0 - 110.0_f64)).exp();
        let below = 46.0 + 23.4 * (-0.036 * (150.0 - 110.0_f64)).exp();
        assert_eq!(calc_latn(500.0, 110.0), above);
        assert!((above - 51.277_142_808_079_475).abs() < 1e-12);
        assert_eq!(calc_latn(0.0, 110.0), below);
        assert_eq!(calc_latn(-5.0, 110.0), below);
        assert!(calc_latn(500.0, f64::NAN).is_nan());
        assert!(calc_latn(f64::NAN, 110.0).is_nan());
    }

    #[test]
    fn d0_is_a_whole_day_from_the_elevation_branch_and_missing_where_an_input_is() {
        let latn = calc_latn(500.0, 110.0);
        assert_eq!(calc_d0(55.0, latn, 500.0), 161.0);
        let latn0 = calc_latn(0.0, 110.0);
        assert_eq!(
            calc_d0(55.0, latn0, 0.0),
            (151.0 * (55.0 / latn0)).round_ties_even()
        );
        assert!(calc_d0(f64::NAN, latn, 500.0).is_nan());
        assert!(calc_d0(55.0, f64::NAN, 500.0).is_nan());
        // not limited to a calendar day
        assert_eq!(calc_d0(83.0, calc_latn(5959.0, 52.0), 5959.0), 370.0);
    }

    #[test]
    fn dj_is_the_day_of_year_zero_for_an_infinite_latn_and_nan_for_a_missing_one() {
        assert_eq!(calc_dj(51.0, 20_230_615), 166.0);
        assert_eq!(calc_dj(51.0, 20_240_301), 61.0);
        assert_eq!(calc_dj(f64::INFINITY, 20_230_615), 0.0);
        assert!(calc_dj(f64::NAN, 20_230_615).is_nan());
        assert!(calc_dj(51.0, 20_230_231).is_nan());
    }

    #[test]
    fn nd_is_the_distance_in_days_whatever_the_side() {
        assert_eq!(calc_nd(100.0, 150.0), 50.0);
        assert_eq!(calc_nd(200.0, 150.0), 50.0);
        assert!(calc_nd(f64::NAN, 150.0).is_nan());
    }

    #[test]
    fn fme_follows_the_formula_and_a_missing_fmc() {
        assert!((calc_fme(120.0) - 0.525_192_603_699_551_5).abs() < 1e-14);
        assert!((calc_fme(100.0) - 0.738_319_800_204_918_3).abs() < 1e-14);
        assert!(calc_fme(f64::NAN).is_nan());
    }

    #[test]
    fn the_orchestrator_chains_the_functions_and_honours_explicit_overrides() {
        let r = calc_foliar_moisture(55.0, 110.0, 500.0, 20_230_615, None, None, None);
        assert_eq!((r.d0, r.dj, r.nd), (161.0, 166.0, 5.0));
        assert!((r.fmc - 85.4725).abs() < 1e-12);
        assert_eq!(r.fme, calc_fme(r.fmc));
        let o = calc_foliar_moisture(
            55.0,
            110.0,
            500.0,
            20_230_615,
            Some(150.0),
            Some(170.0),
            None,
        );
        assert_eq!((o.d0, o.dj, o.nd), (150.0, 170.0, 20.0));
        let missing =
            calc_foliar_moisture(55.0, 110.0, 500.0, 20_230_615, Some(f64::NAN), None, None);
        assert!(missing.nd.is_nan() && missing.fmc.is_nan() && missing.fme.is_nan());
    }

    #[test]
    fn a_negative_override_asks_for_the_derived_value_and_nan_stays_missing() {
        let base = calc_foliar_moisture(55.0, 110.0, 500.0, 20_230_615, None, None, None);
        for negative in [-1.0, -5.0, -0.5, -999.0] {
            let r = calc_foliar_moisture(
                55.0,
                110.0,
                500.0,
                20_230_615,
                Some(negative),
                Some(negative),
                Some(negative),
            );
            assert_eq!((r.d0, r.dj, r.nd), (base.d0, base.dj, base.nd));
            assert_eq!((r.fmc, r.fme), (base.fmc, base.fme));
        }
        // a calculate request beside a given value: each is its own cell's business
        let r = calc_foliar_moisture(
            55.0,
            110.0,
            500.0,
            20_230_615,
            Some(-1.0),
            Some(170.0),
            None,
        );
        assert_eq!((r.d0, r.dj), (base.d0, 170.0));
        let r = calc_foliar_moisture(
            55.0,
            110.0,
            500.0,
            20_230_615,
            Some(f64::NAN),
            None,
            Some(-1.0),
        );
        assert!(
            r.d0.is_nan() && r.nd.is_nan() && r.fmc.is_nan(),
            "a NaN d0 is missing, not calculated"
        );
    }
}
