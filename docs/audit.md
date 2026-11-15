# Audit of the Original Implementation

This audit covers the project as merged into `main` at commit `7e166f1` (five scripts in `src/`). Line numbers refer to that commit.
Every claim marked **Confirmed** was checked by running code. Items marked **Risk** need information that the public data does not provide.

## 1. Current implementation overview

| Step | File | What it did |
|---|---|---|
| Explore | `src/explore.py` | Printed checks of the raw data (missing values, duplicates, outliers, leakage hints); saved one figure |
| Clean | `src/clean.py` | Removed 850 of 119,390 rows (0.71%) with nine rules; wrote `data/hotel_booking_clean.csv` |
| Describe | `src/visualize.py` | Four descriptive figures |
| Features | `src/features.py` | Removed Non Refund bookings; built `prev_cancel`, `prev_stay`, `kids`, top-10 `agent_group`; split by **arrival date**: train = arrivals 2015–2016, test = arrivals in 2017 |
| Model | `src/model.py` | Logistic regression; terms kept if they lowered training BIC **and** did not raise **test** log loss; reported test log loss 0.525, AUC 0.769, daily correlation 0.916 |

**Baseline run (before any change):** all five scripts ran without errors or warnings in a fresh Python 3.13 environment
(pandas 3.0, statsmodels 0.15) in about 27 seconds, and reproduced the README numbers exactly. The scripts had to be run from
the repository root, because every path was relative (for example `features.py:25`, `clean.py:4`).

## 2. Identified methodological issues

| # | Status | Where | Problem → Cause → Impact → Fix |
|---|---|---|---|
| M1 | **Confirmed** | `model.py:24-38`, `model.py:43-68` | **The test set chose the model.** `test_loss()` scores every candidate on 2017 data, and the selection rule uses that score. → The same data selects terms and reports performance. → Reported test metrics are optimistically biased, and the size of the bias is unknown. → Select on a separate validation window; touch the test set once. |
| M2 | **Confirmed** | `features.py:14`, `features.py:45-46` | **The split uses arrival date, but the stated prediction time is the booking date.** 14,099 of the 40,507 bookings arriving in 2017 (35%) were made in 2016 or earlier, before the model's training data ends. → When those bookings were made, the outcomes of bookings that arrived between their booking date and 2016-12-31 were not yet known. → The test set mixes in look-ahead relative to the prediction time. → Split by reconstructed booking date (see `docs/methodology.md`). |
| M3 | **Confirmed** | README "Results", `model.py:76-78` | **The duplicate check was too narrow.** The check printed only five coefficients and concluded they were "very similar". → Several coefficients change substantially without duplicates (for example `has_company_id` -1.19 → -0.66, agent 6 -0.91 → -0.21, meal FB 1.01 → 0.52). → Individual agent, meal and company effects are not robust to duplicates; predictions are (test AUC 0.733 vs 0.735). → Report the full comparison and state which conclusions are robust. |
| M4 | **Confirmed** | README "Results" | **The single odds ratio for Resort Hotel was presented as a main finding (OR 0.35).** The model has `log1p(lead_time) * hotel`, so that coefficient is the effect at lead time 0 only. → The README described a conditional coefficient as a general effect. → The Resort effect ranges from OR 0.38 at lead time 0 to 0.68 at 365 days. → Report the effect as a function of lead time. |
| M5 | **Confirmed** | `model.py` (whole file) | **No baseline model was reported.** Only an intercept-only log loss was printed. → Readers could not tell how much the model adds. → Add training-period overall and hotel-specific rate baselines and a ranking comparison. |
| M6 | **Confirmed** | `model.py:113-120` | **The daily forecast was reported only as a correlation.** Daily totals follow booking volume, so any volume-based forecast correlates well. → A hotel-rate × booking-count baseline already reaches a correlation of 0.88. → Compare MAE, RMSE and bias against that baseline. |
| M7 | Risk | `features.py:26` | **Non Refund bookings were removed without documenting the scope.** → The model silently does not cover 12% of bookings. → Compare a single model, an exclusion and a segmented scheme; state the scope. |

## 3. Data leakage assessment

| Field | Evidence | Status | Original handling |
|---|---|---|---|
| `reservation_status`, `reservation_status_date` | Status equals the outcome by definition | **Confirmed** leakage | Excluded ✓ |
| `assigned_room_type` | Cancellation rate 5.4% when it differs from the reserved room vs 41.6% when equal; rooms are assigned at check-in | **Confirmed** leakage | Excluded ✓ |
| `required_car_parking_spaces` | 0% cancellation in 7,390 bookings with any parking space: recorded at arrival | **Confirmed** leakage | Excluded ✓ |
| `booking_changes` | 15.8% cancellation with changes vs 41.0% without; changes accumulate until arrival or cancellation | Risk | Excluded ✓ |
| `total_of_special_requests` | 48.0% cancellation with 0 requests vs about 22% with 1 or more; requests can be added after booking | Risk | **Used in the final model** |
| `country` | Missing country cancels at 12.1% vs 37.4%; the source paper notes nationality is often confirmed at check-in | Risk | Not used ✓ |
| `previous_cancellations` | The effect reverses over time (99.5% cancellation in 2015Q3 vs 8.9% in 2017Q1) | Risk: unclear whether counted at booking time | Rejected by selection ✓ |
| `adr` | ADR may reflect the final stay charges for checked-out bookings | Risk | Rejected by selection ✓ |

A full feature availability audit with the final decisions is in [`methodology.md`](methodology.md#2-feature-availability-audit).

## 4. Modeling and evaluation issues

| # | Status | Where | Issue | Fix |
|---|---|---|---|---|
| E1 | **Confirmed** | `features.py:40-41` | Category levels were taken from the full data set (including test rows), and a level only seen in the test set would create an all-zero training column. No such level occurred, so results were not affected. | Levels learned on the training window; rare or unseen levels become `Other` |
| E2 | **Confirmed** | `model.py:85-110` | Only log loss, AUC and a decile calibration plot were reported; no Average Precision, Brier score, calibration slope, CIs, ranking or subgroup metrics | Full metric set with day-clustered bootstrap CIs |
| E3 | **Confirmed** | `model.py` | No uncertainty for any test metric | Bootstrap that resamples arrival dates (bookings on one date are correlated, e.g. group blocks) |
| E4 | Risk | `model.py:16-17` | The interaction `lead_time × hotel` was kept by the BIC/test rule but had no stated rationale | Kept on validation evidence; the empirical log-odds figure shows different slopes |
| E5 | **Confirmed** | README | No link from predictions to decisions | Risk tiers, Top-K capture and a scenario-based intervention model |

## 5. Engineering and reproducibility issues

| # | Status | Where | Issue | Fix |
|---|---|---|---|---|
| R1 | **Confirmed** | all scripts | Paths are relative to the working directory | `pathlib` paths in `src/config.py` |
| R2 | **Confirmed** | all scripts | Code runs on import, with no functions, so nothing can be unit tested | Functions plus `main()`; one entry point `python -m src.run_pipeline` |
| R3 | **Confirmed** | `features.py:14`, `model.py:16` | Settings are scattered (split date, formula, top-10) | One `config.py` |
| R4 | **Confirmed** | repository | No tests, no CI, unpinned dependencies, no Python version | pytest suite, GitHub Actions, version ranges in `pyproject.toml` |
| R5 | **Confirmed** | repository | Results existed only as printed output | Machine-readable files in `results/` |

## 6. Proposed solutions and priority

| Priority | Solution | Addresses |
|---|---|---|
| P0 | Booking-date temporal design with train / validation / test; selection on validation only | M1, M2 |
| P0 | Feature availability audit with strict and extended feature sets | Leakage risks |
| P0 | Tests for split boundaries, overlap, training-only encoding, leakage-free feature sets | M1, M2, E1 |
| P1 | Baselines, full metrics, bootstrap CIs, Top-K, subgroups, daily baseline | M5, M6, E2, E3 |
| P1 | Interaction effects by lead time; full duplicate robustness | M3, M4 |
| P1 | Non Refund scheme comparison; cleaning report | M7 |
| P1 | Scenario-based business simulation | E5 |
| P2 | Package structure, config, CI, results files, documentation | R1–R5 |

## 7. Existing results worth preserving

- **The cleaning rules**: they remove only 0.71% of rows and change the cancellation rate by 0.2 points (37.0% → 37.3%). They are kept, with each rule now classified and counted (`results/data_cleaning_report.csv`).
- **The selection rule "lower BIC and better out-of-sample log loss"**: sound; only the out-of-sample set changes (validation instead of test).
- **`log1p(lead_time)`, `prev_stay`, `kids`, top-10 agent grouping, reference levels**: kept.
- **Rejecting `prev_cancel` because its effect drifts**: confirmed again on the validation window (BIC -5,058 but validation log loss +0.008).
- **Exclusion of the clearly leaking fields**: kept and now enforced by a test.
- **Descriptive figures and the odds ratio plot**: kept and regenerated.
- **Continuity check**: the new code reproduces the original headline result exactly (original specification, 2017 arrivals: AUC 0.7685, log loss 0.5252), so the change in reported performance can be traced step by step (see the README).

## 8. Second review (final refinement)

A focused review of the refactored version found no defect in the splits or the model code. The model outputs are unchanged.
The changes concern interpretation and presentation.

| # | Where | Issue | Resolution |
|---|---|---|---|
| F1 | `src/business.py` `simulate_intervention`, README | `realized_*` results (observed test labels × assumed effects) were named and presented as if they were realized profit, and led the README | Renamed to `outcome_conditioned_*`; expected (ex-ante) values are now the primary result; both views are labeled in tables and the figure |
| F2 | `src/business.py`, `src/run_pipeline.py` | No decision rule independent of test outcomes; coverage curves could be read as test-optimized policies | Added the threshold rule "contact if expected profit > 0" (no free parameter), checked on validation, applied unchanged to test (`results/business_policy.csv`); coverage curves labeled as comparative scenario analysis |
| F3 | README | The p × V ranking was presented without its justification | Documented that it equals the expected-profit ranking when e, r, m and c are constant, and added a test |
| F4 | README | Company ID and full board were listed as risk factors although they are not robust to deduplication | Pre-declared robustness rule (same sign, change ≤ 0.2 log-odds) with a `robust_to_dedup` column; the README reports robust and non-robust coefficients separately |
| F5 | `src/features.py` `encode` | Unseen levels fall back to the reference level when no `Other` level exists | Kept, because an empty `Other` level would be unidentifiable. Documented that the reference is the modal level; added `results/encoding_fallbacks.csv` and a test showing the fallback is never triggered in this data |
| F6 | README, methodology | Test-population truncation (lead time ≤ 241 days) appeared only in the limitations | Lead-time coverage per split added to `results/split_summary.csv` and stated next to the main results |
| F7 | methodology, limitations | Bootstrap limitations were not stated | Documented what date-clustered resampling does and does not cover |
| F8 | README | Wording such as "only information available at booking" overstated certainty | Rephrased as "judged likely to be available at booking time", with the residual leakage risk stated |
