# Limitations

**Data scope**
- Two Portuguese hotels, arrivals 2015-07 to 2017-08. Results do not transfer automatically to other hotels, markets or years; booking behavior and distribution channels have changed since then.
- Only the final state of each booking is recorded. There is no history of changes, so point-in-time features cannot be rebuilt.
- The booking date is reconstructed from lead time and arrival date; it is not an original time stamp.
- The data was selected by arrival date. Bookings made in 2017 for arrivals after August 2017 are missing, so the test set covers lead times of at most 241 days.

**Feature availability**
- When each field was recorded is not documented. The strict model uses fields judged likely to be known at booking, but fields such as meal plan, party size, nights, room type and ADR may have been modified later. The evaluation is out of time, not a replay of historical system states, and some residual leakage risk cannot be removed with this data.
- Unseen category values fall back to `Other` or, for columns without an `Other` level, to the modal reference level. This never happens in the evaluated data, but it would treat a genuinely new category as the most common one.
- The extended set (special requests, booking changes, waiting list) improves test AUC from 0.733 to 0.774, and part of that gain may come from information recorded after booking.

**Dependence in the data**
- 26.9% of cleaned rows are exact duplicates and there is no booking ID. Group bookings and duplicates make observations dependent.
- The bootstrap resamples arrival dates, which covers same-day dependence. It does not cover repeated guests, agents, or adjacent dates, so CIs are likely somewhat too narrow.
- Coefficients of individual agents, the company flag, the full board meal plan and some segments change substantially without duplicates and are not treated as robust insights.

**Non Refund bookings**
- They are handled by a simple rule (training cancellation rate 99.2%). Their unusual pattern (mostly group blocks, very high cancellation) suggests a data or process artifact that the public data cannot explain.

**Validation design**
- One validation window (July–December 2016 bookings) with a different season mix than the test window. Rolling-origin validation was not feasible with 26 months of data.
- Gradient boosting was worse than the strict logistic model on validation but better on test, which shows that model rankings are not stable across periods under drift.
- The previous-cancellation effect reverses over time; other effects can drift as well. A deployed model would need monitoring and periodic refitting.

**Interpretation**
- Logistic regression coefficients are conditional associations, not causal effects. For example, "having a company ID is associated with lower cancellation odds" does not mean that adding a company ID would reduce cancellations.

**Business simulation**
- Intervention effectiveness, cost, margin and replacement probability are assumptions; the data contains no retention experiments. Simulated benefits are illustrative.
- ADR is revenue, not profit; the margin rate is an assumed conversion.
- Not modeled: capacity, overbooking, displacement of other bookings, behavioral changes caused by the contact, revenue retained from deposits, and price effects.
- Expected results use the model's probabilities and are only as reliable as its calibration. Outcome-conditioned results use observed test labels under the same assumptions; they are a retrospective check of the ranking, not realized profit.
- The coverage curves are evaluated on the test set; their maxima are not validated optima. The decision rule used instead (contact if expected profit > 0) is fixed by the assumptions.
