# Limitations

**Data scope**
- Two Portuguese hotels, arrivals 2015-07 to 2017-08. Results do not transfer automatically to other hotels, markets or years; booking behavior and distribution channels have changed since then.
- Only the final state of each booking is recorded. There is no history of changes, so point-in-time features cannot be rebuilt.
- The booking date is reconstructed from lead time and arrival date; it is not an original time stamp.
- The data was selected by arrival date. Bookings made in 2017 for arrivals after August 2017 are missing, so the test set covers lead times of at most 241 days.

**Feature availability**
- When each field was recorded is not documented. The strict model uses fields judged to be known at booking, but fields such as meal plan, party size, nights and room type may have been modified later. The extended set (special requests, booking changes, waiting list) improves test AUC from 0.734 to 0.775, and part of that gain may come from information recorded after booking.

**Dependence in the data**
- 26.9% of cleaned rows are exact duplicates and there is no booking ID. Group bookings and duplicates make observations dependent; the bootstrap resamples arrival dates to account for part of this. Coefficients of individual agents, the company flag and the full board meal plan change substantially without duplicates.

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
- The realized results use the test outcomes of bookings made from January to August 2017; they are a back-test of the ranking under assumed effects, not an estimate of real profit.
