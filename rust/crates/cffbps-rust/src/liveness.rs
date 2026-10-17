//! Which fields a planned set of stages still needs.
//!
//! [`Liveness`] counts, for every [`Field`], the planned stages that read it
//! and have not run yet. The count only depends on which stages are planned
//! and which of them have run at least once, never on the order they ran in,
//! so a caller may run the planned stages in any order. A field whose count is
//! zero has no planned reader left: [`FbpGrid`](crate::fbp_grid::FbpGrid)
//! drops it (or never stores it) unless the caller asked to keep it.

use crate::field::Field;
use crate::stage::Stage;

/// The planned stages, which of them have run, and how many planned readers
/// each field still has.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Liveness {
    planned: [bool; Stage::COUNT],
    ran: [bool; Stage::COUNT],
    pending_readers: [u8; Field::COUNT],
}

impl Liveness {
    /// The bookkeeping for `planned` (duplicates and order do not matter),
    /// with no stage run yet.
    pub(crate) fn new(planned: &[Stage]) -> Self {
        let mut l = Self {
            planned: [false; Stage::COUNT],
            ran: [false; Stage::COUNT],
            pending_readers: [0; Field::COUNT],
        };
        l.replan(planned);
        l
    }

    /// Replace the planned set, keeping the record of which stages have run:
    /// a field's count becomes the number of newly planned stages that read
    /// it and have not run yet.
    pub(crate) fn replan(&mut self, planned: &[Stage]) {
        self.planned = [false; Stage::COUNT];
        for s in planned {
            self.planned[s.index()] = true;
        }
        self.recount();
    }

    /// Forget which stages have run (the counts start again from the plan).
    pub(crate) fn restart(&mut self) {
        self.ran = [false; Stage::COUNT];
        self.recount();
    }

    /// The counts from the planned set and the stages that ran.
    fn recount(&mut self) {
        self.pending_readers = [0; Field::COUNT];
        for s in Stage::ALL {
            if self.planned[s.index()] && !self.ran[s.index()] {
                for f in s.reads() {
                    // at most 23 stages read a field, so this never overflows
                    self.pending_readers[f.index()] += 1;
                }
            }
        }
    }

    /// Whether `stage` is in the planned set.
    #[cfg(test)]
    pub(crate) fn is_planned(&self, stage: Stage) -> bool {
        self.planned[stage.index()]
    }

    /// Whether a planned stage that has not run yet reads `field`.
    pub(crate) fn needed(&self, field: Field) -> bool {
        self.pending_readers[field.index()] > 0
    }

    /// The number of planned stages that read `field` and have not run yet.
    #[cfg(test)]
    pub(crate) fn pending_readers(&self, field: Field) -> u8 {
        self.pending_readers[field.index()]
    }

    /// Record that `stage` ran. The first run of a planned stage counts once
    /// for each field it reads; returns the fields whose last planned reader
    /// that was, in [`Stage::reads`] order. A stage outside the plan, or a
    /// repeat run, changes nothing and returns an empty list.
    pub(crate) fn record_run(&mut self, stage: Stage) -> Vec<Field> {
        let k = stage.index();
        if !self.planned[k] || self.ran[k] {
            return Vec::new();
        }
        self.ran[k] = true;
        let mut last = Vec::new();
        for f in stage.reads() {
            let count = &mut self.pending_readers[f.index()];
            *count -= 1;
            if *count == 0 {
                last.push(*f);
            }
        }
        last
    }
}

#[cfg(test)]
mod tests {
    use super::Liveness;
    use crate::field::Field;
    use crate::stage::Stage;

    #[test]
    fn a_two_stage_plan_counts_each_planned_reader() {
        let mut l = Liveness::new(&[Stage::CalcTfc, Stage::CalcHfi]);
        // ffc, wfc, sfc, cfc: CalcTfc; tfc: CalcHfi; hros: CalcTfc and CalcHfi
        for f in [Field::Ffc, Field::Wfc, Field::Sfc, Field::Cfc, Field::Tfc] {
            assert_eq!(l.pending_readers(f), 1, "{f:?}");
            assert!(l.needed(f), "{f:?}");
        }
        assert_eq!(l.pending_readers(Field::Hros), 2);
        assert!(l.needed(Field::Hros));
        assert!(!l.needed(Field::Hfi), "no planned stage reads hfi");
        let mut freed = l.record_run(Stage::CalcTfc);
        freed.sort_by_key(|f| f.index());
        let mut expected = [Field::Ffc, Field::Wfc, Field::Sfc, Field::Cfc];
        expected.sort_by_key(|f| f.index());
        assert_eq!(freed, expected);
        assert!(l.needed(Field::Hros), "calcHFI still reads hros");
        assert!(l.needed(Field::Tfc));
        assert_eq!(l.record_run(Stage::CalcHfi), [Field::Hros, Field::Tfc]);
        for f in Field::ALL {
            assert!(!l.needed(f), "{f:?}");
        }
    }

    #[test]
    fn the_full_plan_counts_every_reader_and_ends_with_none() {
        let mut l = Liveness::new(&Stage::ALL);
        for f in Field::ALL {
            let readers = Stage::ALL.iter().filter(|s| s.reads().contains(&f)).count();
            assert_eq!(usize::from(l.pending_readers(f)), readers, "{f:?}");
        }
        // hros: calcC6HROS, calcPercentileCFB, calcRosPercentileGrowth, calcCFB, calcTFC, calcHFI
        assert_eq!(l.pending_readers(Field::Hros), 6);
        let mut freed_total = 0;
        for s in Stage::ALL {
            freed_total += l.record_run(s).len();
        }
        let read_fields = Field::ALL
            .iter()
            .filter(|f| Stage::ALL.iter().any(|s| s.reads().contains(f)))
            .count();
        assert_eq!(
            freed_total, read_fields,
            "each read field is freed exactly once"
        );
        for f in Field::ALL {
            assert_eq!(l.pending_readers(f), 0, "{f:?}");
        }
    }

    #[test]
    fn a_field_read_by_no_stage_is_never_needed() {
        let l = Liveness::new(&Stage::ALL);
        for f in [Field::Raz, Field::FiClass, Field::BrosCfb, Field::Latn] {
            assert!(!l.needed(f), "{f:?}");
        }
    }

    #[test]
    fn a_field_read_twice_is_freed_after_its_second_reader() {
        // sfc is read by calcRSO and calcTFC
        let mut l = Liveness::new(&[Stage::CalcRso, Stage::CalcTfc]);
        assert_eq!(l.pending_readers(Field::Sfc), 2);
        assert!(!l.record_run(Stage::CalcRso).contains(&Field::Sfc));
        assert!(l.needed(Field::Sfc));
        assert!(l.record_run(Stage::CalcTfc).contains(&Field::Sfc));
        assert!(!l.needed(Field::Sfc));
    }

    #[test]
    fn a_repeat_run_or_an_unplanned_stage_changes_nothing() {
        let mut l = Liveness::new(&[Stage::CalcRso, Stage::CalcTfc]);
        l.record_run(Stage::CalcRso);
        let before = l.clone();
        assert!(l.record_run(Stage::CalcRso).is_empty(), "repeat");
        assert!(l.record_run(Stage::CalcHfi).is_empty(), "unplanned");
        assert_eq!(l, before);
        assert!(!l.is_planned(Stage::CalcHfi));
        assert!(l.is_planned(Stage::CalcTfc));
    }

    #[test]
    fn the_run_order_does_not_matter() {
        let mut forward = Liveness::new(&Stage::ALL);
        let mut backward = Liveness::new(&Stage::ALL);
        let mut freed_forward = Vec::new();
        let mut freed_backward = Vec::new();
        for s in Stage::ALL {
            freed_forward.extend(forward.record_run(s));
        }
        for s in Stage::ALL.iter().rev() {
            freed_backward.extend(backward.record_run(*s));
        }
        assert_eq!(forward, backward);
        freed_forward.sort_by_key(|f| f.index());
        freed_backward.sort_by_key(|f| f.index());
        assert_eq!(freed_forward, freed_backward);
        // a plan listed in another order is the same plan
        let mut reversed: Vec<Stage> = Stage::ALL.to_vec();
        reversed.reverse();
        assert_eq!(Liveness::new(&reversed), Liveness::new(&Stage::ALL));
        assert_eq!(
            Liveness::new(&[Stage::CalcHfi, Stage::CalcHfi]),
            Liveness::new(&[Stage::CalcHfi])
        );
    }

    #[test]
    fn a_skipped_planned_stage_keeps_its_inputs_needed() {
        let mut l = Liveness::new(&Stage::ALL);
        for s in Stage::ALL {
            if s != Stage::CalcFmc {
                l.record_run(s);
            }
        }
        for f in Stage::CalcFmc.reads() {
            assert!(l.needed(*f), "{f:?}");
        }
        // an exact plan without the stage releases them
        l.replan(
            &Stage::ALL
                .iter()
                .copied()
                .filter(|s| *s != Stage::CalcFmc)
                .collect::<Vec<_>>(),
        );
        for f in Field::ALL {
            assert!(!l.needed(f), "{f:?}");
        }
    }

    #[test]
    fn restart_counts_from_the_plan_again() {
        let fresh = Liveness::new(&Stage::ALL);
        let mut l = fresh.clone();
        for s in Stage::ALL {
            l.record_run(s);
        }
        l.restart();
        assert_eq!(l, fresh);
    }
}
