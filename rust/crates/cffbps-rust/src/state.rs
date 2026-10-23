//! A per-cell state: one value per [`Field`], and the stage bodies that read
//! and write it.
//!
//! [`CellState::apply`] runs one [`Stage`]. Each stage body is the
//! corresponding step of the scalar pass, reading only the fields in
//! [`Stage::reads`] and writing only the fields in [`Stage::writes`], so a
//! state can be advanced stage by stage, inspected or changed between stages
//! with [`CellState::get`] / [`CellState::set`], and converted to an
//! [`FbpResult`] at any point. Running [`Stage::ALL`] in order on
//! [`CellState::from_input`] is [`fbp::run`](crate::fbp::run).

use crate::behavior::{calc_fire_intensity_class, calc_fire_type, calc_hfi};
use crate::consumption::{calc_cfc, calc_sfc, calc_tfc, without_spread, SurfaceFuel};
use crate::crown::{calc_csfi, calc_rso, cfb_from_ros, directional_cfb, final_cfb};
use crate::fbp::{FbpInput, FbpResult};
use crate::field::Field;
use crate::foliar_moisture::{calc_foliar_moisture, FoliarMoisture};
use crate::fuel::{CrownFuel, FuelType, RosParams};
use crate::growth::calc_accel_param;
use crate::normalize::{invert_wind_aspect, normalize, Normalized};
use crate::percentile::{percentile_ros, percentile_tinv, wind_decay};
use crate::ros::{c6_cros, c6_hros, calc_ros, Ros};
use crate::slope_wind::{calc_isi_rsi_be, calc_isz, calc_sf, Isz, SpreadIndices};
use crate::stage::Stage;

/// The scalar (not per-field) inputs of the stages.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StageParams {
    /// Date as `YYYYMMDD`, read by `calcFMC`.
    pub wx_date: i64,
    /// Percentile of the ROS distribution (50 is the unadjusted ROS), read by
    /// `calcRosPercentileGrowth`.
    pub percentile_growth: f64,
    /// Whether an explicit `d0` is supplied, and its initial value: `Some`
    /// makes `calcFMC` use the [`Field::D0Override`] field (NaN = masked)
    /// instead of deriving `d0`; `None` derives it.
    pub d0: Option<f64>,
    /// As `d0`, for `dj` and [`Field::DjOverride`].
    pub dj: Option<f64>,
    /// As `d0`, for `fmc` and [`Field::FmcOverride`].
    pub fmc: Option<f64>,
}

/// The values of every [`Field`] for one cell, plus the scalar
/// [`StageParams`].
///
/// [`CellState::set`] stores a value as given; it has none of the side effects
/// the grid-level API gives some fields (such as clearing the `hros` growth
/// provenance or redirecting `wd` to `wd_in`), so it can also poison any field
/// for the stage tests. `set(Field::FuelType, v)` changes the fuel the stages
/// use (`v` truncated to an integer code).
#[derive(Clone, Debug)]
pub struct CellState {
    values: [f64; Field::COUNT],
    ft: FuelType,
    params: StageParams,
    /// `FbpInput::hros_override`: `calcROS` replaces its head ROS with it.
    hros_injection: Option<f64>,
}

impl CellState {
    /// A state with every input missing (NaN) and every computed field at its
    /// [`Field::template_value`], for fuel code `fuel_type`. The explicit
    /// `d0` / `dj` / `fmc` values of `params`, if any, are stored in
    /// [`Field::D0Override`] / [`Field::DjOverride`] / [`Field::FmcOverride`].
    #[must_use]
    pub fn template(fuel_type: i32, params: StageParams) -> Self {
        let mut values = [0.0; Field::COUNT];
        for f in Field::ALL {
            values[f.index()] = f.template_value();
        }
        let mut s = Self {
            values,
            ft: FuelType::from_code(fuel_type),
            params,
            hros_injection: None,
        };
        s.set(Field::FuelType, f64::from(fuel_type));
        s.set(Field::D0Override, params.d0.unwrap_or(f64::NAN));
        s.set(Field::DjOverride, params.dj.unwrap_or(f64::NAN));
        s.set(Field::FmcOverride, params.fmc.unwrap_or(f64::NAN));
        s
    }

    /// The state [`fbp::run`](crate::fbp::run) starts from: the template with
    /// the inputs of `input`, normalised as `FBP.initialize` does (`long` is
    /// stored as its magnitude; `wd` and `aspect` hold the inputs until
    /// `invertWindAspect` runs). `hros_override` replaces `calcROS`'s head ROS, as in [`FbpInput`].
    #[must_use]
    pub fn from_input(input: &FbpInput) -> Self {
        let mut s = Self::template(
            input.fuel_type,
            StageParams {
                wx_date: input.wx_date,
                percentile_growth: input.percentile_growth,
                d0: input.d0_override,
                dj: input.dj_override,
                fmc: input.fmc_override,
            },
        );
        let n = normalize(input);
        for (f, v) in [
            (Field::Lat, input.lat),
            (Field::Long, input.long.abs()),
            (Field::Elevation, input.elevation),
            (Field::Slope, n.slope),
            (Field::AspectIn, n.aspect),
            (Field::Aspect, n.aspect),
            (Field::Ws, n.ws),
            (Field::WdIn, n.wd),
            (Field::Wd, n.wd),
            (Field::Ffmc, n.ffmc),
            (Field::Bui, n.bui),
            (Field::Pc, n.pc),
            (Field::Pdf, n.pdf),
            (Field::Gfl, n.gfl),
            (Field::Gcf, n.gcf),
        ] {
            s.set(f, v);
        }
        s.hros_injection = input.hros_override;
        s
    }

    /// The current value of `f`.
    #[must_use]
    pub fn get(&self, f: Field) -> f64 {
        self.values[f.index()]
    }

    /// Store `v` in `f`.
    pub fn set(&mut self, f: Field, v: f64) {
        self.values[f.index()] = v;
        if f == Field::FuelType {
            // Saturating cast (NaN gives 0): an integral code converts exactly.
            #[allow(clippy::cast_possible_truncation)]
            let code = v as i32;
            self.ft = FuelType::from_code(code);
        }
    }

    /// The scalar parameters.
    #[must_use]
    pub fn params(&self) -> &StageParams {
        &self.params
    }

    /// Normalised inputs as `calc_isi_rsi_be` / `calc_sfc` take them. Only the
    /// fields named by `with` are read from the state; the rest are NaN, and
    /// the callers below do not read them.
    fn normalized(&self, with: &[Field]) -> Normalized {
        let g = |f: Field| {
            if with.contains(&f) {
                self.get(f)
            } else {
                f64::NAN
            }
        };
        Normalized {
            slope: g(Field::Slope),
            aspect: g(Field::Aspect),
            ws: g(Field::Ws),
            wd: g(Field::Wd),
            ffmc: g(Field::Ffmc),
            bui: g(Field::Bui),
            pc: g(Field::Pc),
            pdf: g(Field::Pdf),
            gfl: g(Field::Gfl),
            gcf: g(Field::Gcf),
        }
    }

    /// Run one stage: read the fields of [`Stage::reads`], write the fields
    /// of [`Stage::writes`]. Never panics; missing values propagate as NaN.
    // One arm per stage, each the corresponding step of the scalar pass.
    #[allow(clippy::too_many_lines)]
    pub fn apply(&mut self, stage: Stage) {
        use Field as F;
        let ft = self.ft;
        let c6 = ft == FuelType::C6;
        match stage {
            Stage::InvertWindAspect => {
                // always from the inputs, so repeating the stage never flips twice
                let (wd, aspect) = invert_wind_aspect(self.get(F::WdIn), self.get(F::AspectIn));
                self.set(F::Wd, wd);
                self.set(F::Aspect, aspect);
            }
            Stage::CalcSf => {
                let sf = calc_sf(self.get(F::Slope));
                self.set(F::Sf, sf);
            }
            Stage::CalcIsz => {
                let Isz { m, f_f, isz } = calc_isz(self.get(F::Ffmc));
                self.set(F::M, m);
                self.set(F::FF, f_f);
                self.set(F::Isz, isz);
            }
            Stage::CalcFmc => {
                let FoliarMoisture {
                    latn,
                    d0,
                    dj,
                    nd,
                    fmc,
                    fme,
                } = calc_foliar_moisture(
                    self.get(F::Lat),
                    self.get(F::Long),
                    self.get(F::Elevation),
                    self.params.wx_date,
                    self.params.d0.map(|_| self.get(F::D0Override)),
                    self.params.dj.map(|_| self.get(F::DjOverride)),
                    self.params.fmc.map(|_| self.get(F::FmcOverride)),
                );
                for (f, v) in [
                    (F::Latn, latn),
                    (F::D0, d0),
                    (F::Dj, dj),
                    (F::Nd, nd),
                    (F::Fmc, fmc),
                    (F::Fme, fme),
                ] {
                    self.set(f, v);
                }
            }
            Stage::CalcIsiRsiBe => {
                let params = ft.ros_params();
                let n = self.normalized(&[F::Aspect, F::Ws, F::Wd, F::Bui, F::Pc, F::Pdf, F::Gcf]);
                let SpreadIndices {
                    rsz,
                    rsf,
                    isf,
                    sw,
                    rsi,
                    brsi,
                    be,
                } = calc_isi_rsi_be(
                    ft,
                    &params,
                    &n,
                    self.get(F::FF),
                    self.get(F::Isz),
                    self.get(F::Sf),
                );
                let RosParams {
                    a,
                    b,
                    c,
                    q,
                    bui0,
                    be_max,
                } = params;
                for (f, v) in [
                    (F::A, a),
                    (F::B, b),
                    (F::C, c),
                    (F::Q, q),
                    (F::Bui0, bui0),
                    (F::BeMax, be_max),
                    (F::Rsz, rsz),
                    (F::Rsf, rsf),
                    (F::Isf, isf),
                    (F::Rsi, rsi),
                    (F::Brsi, brsi),
                    (F::Be, be),
                    (F::Wse1, sw.wse1),
                    (F::Wse2, sw.wse2),
                    (F::Wse, sw.wse),
                    (F::Wsx, sw.wsx),
                    (F::Wsy, sw.wsy),
                    (F::Wsv, sw.wsv),
                    (F::Raz, sw.raz),
                    (F::FW, sw.f_w),
                    (F::Bfw, sw.bfw),
                    (F::Isi, sw.isi),
                    (F::Bisi, sw.bisi),
                ] {
                    self.set(f, v);
                }
            }
            Stage::CalcRos => {
                let Ros { hros, bros, sros } = calc_ros(
                    ft,
                    self.get(F::Rsi),
                    self.get(F::Brsi),
                    self.get(F::Be),
                    self.get(F::Bui),
                    self.hros_injection,
                );
                // Python writes sros for C-6 cells only and keeps the rest
                let sros = if c6 { sros } else { self.get(F::Sros) };
                self.set(F::Hros, hros);
                self.set(F::Bros, bros);
                self.set(F::Sros, sros);
                self.set(F::HrosGrowthNan, 0.0);
            }
            Stage::CalcSfc => {
                let n = self.normalized(&[F::Ffmc, F::Bui, F::Pc, F::Gfl]);
                let SurfaceFuel { sfc, ffc, wfc } = calc_sfc(ft, &n);
                self.set(F::Ffc, ffc);
                self.set(F::Wfc, wfc);
                self.set(F::Sfc, sfc);
            }
            Stage::GetCbhCfl => {
                let CrownFuel { cbh, cfl } = ft.crown_fuel();
                self.set(F::Cbh, cbh);
                self.set(F::Cfl, cfl);
            }
            Stage::CalcCsfi => {
                let csfi = calc_csfi(ft, self.get(F::Cbh), self.get(F::Fmc));
                self.set(F::Csfi, csfi);
            }
            Stage::CalcRso => {
                let rso = calc_rso(self.get(F::Sfc), self.get(F::Csfi));
                self.set(F::Rso, rso);
            }
            // --- deterministic C-6 blend: SROS-derived CFB -> CFC -> CROS ->
            // blended HROS. This CFB is temporary; it is not the CFB used
            // downstream.
            Stage::CalcC6BlendCfb => {
                let blend_cfb = if c6 {
                    cfb_from_ros(self.get(F::Sros), self.get(F::Rso))
                } else {
                    0.0
                };
                self.set(F::C6BlendCfb, blend_cfb);
            }
            Stage::CalcC6BlendCfc => {
                let blend_cfc = calc_cfc(
                    ft,
                    self.get(F::C6BlendCfb),
                    self.get(F::Cfl),
                    self.get(F::Pc),
                    self.get(F::Pdf),
                );
                self.set(F::C6BlendCfc, blend_cfc);
            }
            Stage::CalcC6Cros => {
                if c6 {
                    let cros = c6_cros(self.get(F::C6BlendCfc), self.get(F::Isi), self.get(F::Fme));
                    self.set(F::Cros, cros);
                }
            }
            Stage::CalcC6Hros => {
                if c6 {
                    let hros = c6_hros(
                        self.get(F::Sros),
                        self.get(F::C6BlendCfb),
                        self.get(F::Cros),
                    );
                    self.set(F::Hros, hros);
                    self.set(F::HrosGrowthNan, 0.0);
                }
            }
            Stage::CalcPercentileCfb => {
                let rso = self.get(F::Rso);
                let percentile_cfb = directional_cfb(ft, self.get(F::Hros), rso);
                let percentile_bros_cfb = directional_cfb(ft, self.get(F::Bros), rso);
                self.set(F::PercentileCfb, percentile_cfb);
                self.set(F::PercentileBrosCfb, percentile_bros_cfb);
            }
            Stage::CalcRosPercentileGrowth => {
                // 50 is an exact no-op; NaN propagates
                let (hros_before, bros_before) = (self.get(F::Hros), self.get(F::Bros));
                let percentile = self.params.percentile_growth;
                let (hros, bros) = if percentile == 50.0 {
                    (hros_before, bros_before)
                } else {
                    let tinv = percentile_tinv(percentile);
                    (
                        percentile_ros(ft, hros_before, self.get(F::PercentileCfb), tinv, 1.0),
                        percentile_ros(
                            ft,
                            bros_before,
                            self.get(F::PercentileBrosCfb),
                            tinv,
                            wind_decay(self.get(F::Wsv)),
                        ),
                    )
                };
                let growth_made_nan = hros.is_nan() && !hros_before.is_nan();
                self.set(F::Hros, hros);
                self.set(F::Bros, bros);
                self.set(F::HrosGrowthNan, if growth_made_nan { 1.0 } else { 0.0 });
            }
            Stage::CalcCfb => {
                // final CFB from the percentile-adjusted head ROS (NaN rule in
                // final_cfb)
                let rso = self.get(F::Rso);
                let growth_made_nan = self.get(F::HrosGrowthNan) != 0.0;
                let cfb = final_cfb(ft, self.get(F::Hros), growth_made_nan, rso);
                let bros_cfb = directional_cfb(ft, self.get(F::Bros), rso);
                self.set(F::Cfb, cfb);
                self.set(F::BrosCfb, bros_cfb);
            }
            Stage::CalcAccelParam => {
                // Python writes modeled fuels only and keeps the rest
                let accel = if ft.is_modeled() {
                    calc_accel_param(ft, self.get(F::Cfb))
                } else {
                    self.get(F::Accel)
                };
                self.set(F::Accel, accel);
            }
            Stage::CalcFireType => {
                let fire_type = calc_fire_type(ft, self.get(F::Cfb));
                self.set(F::FireType, fire_type);
            }
            Stage::CalcCfc => {
                let cfc = calc_cfc(
                    ft,
                    self.get(F::Cfb),
                    self.get(F::Cfl),
                    self.get(F::Pc),
                    self.get(F::Pdf),
                );
                self.set(F::Cfc, cfc);
            }
            Stage::CalcTfc => {
                let hros = self.get(F::Hros);
                for f in [F::Ffc, F::Wfc, F::Sfc, F::Cfc] {
                    let adjusted = without_spread(hros, self.get(f));
                    self.set(f, adjusted);
                }
                let tfc = calc_tfc(self.get(F::Sfc), self.get(F::Cfc));
                self.set(F::Tfc, tfc);
            }
            Stage::CalcHfi => {
                let hfi = calc_hfi(self.get(F::Hros), self.get(F::Tfc));
                self.set(F::Hfi, hfi);
            }
            Stage::CalcFireIntensityClass => {
                let fi_class = calc_fire_intensity_class(self.get(F::Hfi));
                self.set(F::FiClass, fi_class);
            }
        }
    }

    /// The 54 outputs as an [`FbpResult`].
    #[must_use]
    pub fn into_result(self) -> FbpResult {
        use Field as F;
        let g = |f: Field| self.get(f);
        FbpResult {
            ws: g(F::Ws),
            wd: g(F::Wd),
            wse: g(F::Wse),
            wse1: g(F::Wse1),
            wse2: g(F::Wse2),
            wsx: g(F::Wsx),
            wsy: g(F::Wsy),
            wsv: g(F::Wsv),
            raz: g(F::Raz),
            m: g(F::M),
            f_f: g(F::FF),
            f_w: g(F::FW),
            ffmc: g(F::Ffmc),
            isi: g(F::Isi),
            bui: g(F::Bui),
            a: g(F::A),
            b: g(F::B),
            c: g(F::C),
            q: g(F::Q),
            bui0: g(F::Bui0),
            be: g(F::Be),
            be_max: g(F::BeMax),
            sf: g(F::Sf),
            rsz: g(F::Rsz),
            rsf: g(F::Rsf),
            isf: g(F::Isf),
            rsi: g(F::Rsi),
            latn: g(F::Latn),
            dj: g(F::Dj),
            d0: g(F::D0),
            nd: g(F::Nd),
            fmc: g(F::Fmc),
            fme: g(F::Fme),
            ffc: g(F::Ffc),
            wfc: g(F::Wfc),
            sfc: g(F::Sfc),
            cfl: g(F::Cfl),
            cfc: g(F::Cfc),
            tfc: g(F::Tfc),
            cbh: g(F::Cbh),
            csfi: g(F::Csfi),
            rso: g(F::Rso),
            cfb: g(F::Cfb),
            fire_type: g(F::FireType),
            hros: g(F::Hros),
            sros: g(F::Sros),
            cros: g(F::Cros),
            bfw: g(F::Bfw),
            bisi: g(F::Bisi),
            bros: g(F::Bros),
            hfi: g(F::Hfi),
            fi_class: g(F::FiClass),
            accel: g(F::Accel),
            fuel_type: g(F::FuelType),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> StageParams {
        StageParams {
            wx_date: 20_230_615,
            percentile_growth: 50.0,
            d0: None,
            dj: None,
            fmc: None,
        }
    }

    #[test]
    fn get_returns_what_set_stored() {
        let mut s = CellState::template(2, params());
        for (k, f) in Field::ALL.into_iter().enumerate() {
            if f != Field::FuelType {
                s.set(f, f64::from(u32::try_from(k).unwrap()) + 0.5);
            }
        }
        for (k, f) in Field::ALL.into_iter().enumerate() {
            if f != Field::FuelType {
                assert_eq!(
                    s.get(f),
                    f64::from(u32::try_from(k).unwrap()) + 0.5,
                    "{f:?}"
                );
            }
        }
    }

    #[test]
    fn template_stores_explicit_overrides() {
        let s = CellState::template(
            2,
            StageParams {
                d0: Some(140.0),
                dj: Some(f64::NAN),
                ..params()
            },
        );
        assert_eq!(s.get(Field::D0Override), 140.0);
        assert!(s.get(Field::DjOverride).is_nan());
        assert_eq!(s.params().d0, Some(140.0));
    }

    #[test]
    fn the_growth_stage_records_a_nan_it_made() {
        let mut s = CellState::template(
            2,
            StageParams {
                percentile_growth: f64::NAN,
                ..params()
            },
        );
        s.set(Field::Hros, 10.0);
        s.set(Field::Bros, 1.0);
        s.set(Field::PercentileCfb, 0.0);
        s.set(Field::PercentileBrosCfb, 0.0);
        s.set(Field::Wsv, 10.0);
        s.apply(Stage::CalcRosPercentileGrowth);
        assert!(s.get(Field::Hros).is_nan());
        assert_eq!(s.get(Field::HrosGrowthNan), 1.0);
        s.set(Field::Rso, 2.0);
        s.apply(Stage::CalcCfb);
        assert_eq!(s.get(Field::Cfb), 0.0, "an unmasked NaN is zeroed");
    }

    #[test]
    fn an_injected_nan_head_ros_gives_a_nan_cfb() {
        let mut s = CellState::template(2, params());
        s.set(Field::Hros, f64::NAN);
        s.set(Field::Rso, 2.0);
        s.apply(Stage::CalcCfb);
        assert!(s.get(Field::Cfb).is_nan());
    }

    #[test]
    fn c6_stages_leave_other_fuels_at_their_current_values() {
        let mut s = CellState::template(2, params());
        s.set(Field::Cros, 3.0);
        s.set(Field::Hros, 4.0);
        s.set(Field::Sros, 5.0);
        for st in [
            Stage::CalcC6BlendCfb,
            Stage::CalcC6BlendCfc,
            Stage::CalcC6Cros,
            Stage::CalcC6Hros,
        ] {
            s.apply(st);
        }
        assert_eq!(s.get(Field::Cros), 3.0);
        assert_eq!(s.get(Field::Hros), 4.0);
        assert_eq!(s.get(Field::C6BlendCfb), 0.0);
    }
}
