//! Foliar moisture content (`fmc.py`) and the calendar helpers it needs.

fn is_leap_year(y: i64) -> bool {
    y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)
}

/// Days in each month of a non-leap year.
const MONTH_DAYS: [i64; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

/// Whether `wx_date` is a real `YYYYMMDD` calendar date (year 1-9999), as
/// `datetime.strptime(str(wx_date), '%Y%m%d')` requires. Callers should check
/// this once before a grid pass; the core itself treats an invalid date as
/// missing (NaN) rather than panicking.
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

/// Day of year from YYYYMMDD — `datetime.strptime(...).timetuple().tm_yday`.
/// NaN for an invalid date.
fn day_of_year(wx_date: i64) -> f64 {
    if !is_valid_wx_date(wx_date) {
        return f64::NAN;
    }
    let y = wx_date / 10_000;
    let mth = (wx_date / 100 % 100) as usize;
    let d = wx_date % 100;
    const CUM: [i64; 12] = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    let mut doy = CUM[mth - 1] + d;
    if mth > 2 && is_leap_year(y) {
        doy += 1;
    }
    doy as f64
}

pub(crate) struct Fmc {
    pub latn: f64,
    pub d0: f64,
    pub dj: f64,
    pub nd: f64,
    pub fmc: f64,
    pub fme: f64,
}

/// `fmc.calc_fmc`
pub(crate) fn calc_fmc(
    lat: f64,
    abs_long: f64,
    elevation: f64,
    wx_date: i64,
    d0_override: Option<f64>,
    dj_override: Option<f64>,
) -> Fmc {
    // Missing geography is a masked cell in Python and stays NaN here: a NaN
    // elevation or longitude masks latn, a NaN latitude masks d0.
    let latn = if elevation.is_nan() || abs_long.is_nan() {
        f64::NAN
    } else if elevation > 0.0 {
        43.0 + 33.7 * (-0.0351 * (150.0 - abs_long)).exp()
    } else {
        46.0 + 23.4 * (-0.036 * (150.0 - abs_long)).exp()
    };
    // rounded to mimic the cffdrs R package (numpy round = ties to even)
    let d0 = match d0_override {
        Some(v) => v,
        None if lat.is_nan() || latn.is_nan() => f64::NAN,
        None => if elevation > 0.0 {
            142.1 * (lat / latn) + 0.0172 * elevation
        } else {
            151.0 * (lat / latn)
        }
        .round_ties_even(),
    };
    let dj = match dj_override {
        Some(v) => v,
        None => {
            if latn.is_nan() {
                f64::NAN
            } else if latn.is_finite() {
                day_of_year(wx_date)
            } else {
                0.0
            }
        }
    };
    let nd = (dj - d0).abs();
    let fmc = if nd.is_nan() {
        f64::NAN
    } else if nd < 30.0 {
        85.0 + 0.0189 * nd * nd
    } else if nd < 50.0 {
        32.9 + 3.17 * nd - 0.0288 * nd * nd
    } else {
        120.0
    };
    let fme = 1000.0 * (1.5 - 0.00275 * fmc).powi(4) / (460.0 + 25.9 * fmc);
    Fmc {
        latn,
        d0,
        dj,
        nd,
        fmc,
        fme,
    }
}

/// The `setParams({'fmc': ...})` injection in place of [`calc_fmc`]:
/// calcFMC never runs, so latn/d0/dj/nd/fme keep their zero templates —
/// the zero fme is load-bearing for C-6, see `FbpInput::fmc_override`.
pub(crate) fn injected_fmc(fmc: f64) -> Fmc {
    Fmc {
        latn: 0.0,
        d0: 0.0,
        dj: 0.0,
        nd: 0.0,
        fmc,
        fme: 0.0,
    }
}
