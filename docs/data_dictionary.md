# Data Dictionary

Source: [Hotel Booking Demand on Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand), from
Antonio, N., de Almeida, A., & Nunes, L. (2019). [Hotel booking demand datasets](https://doi.org/10.1016/j.dib.2018.11.126). *Data in Brief*, 22, 41–49.
Two hotels in Portugal (a City Hotel in Lisbon and a Resort Hotel in the Algarve), arrivals from 2015-07-01 to 2017-08-31. One row per booking; there is no booking ID.

"Use" refers to the models in this project: **S** = strict booking-time set, **E** = extended set only, **–** = not a feature (see the [feature availability audit](methodology.md#2-feature-availability-audit)).

## Columns of `data/hotel_booking_clean.csv`

| Column | Meaning | Use |
|---|---|---|
| `hotel` | City Hotel or Resort Hotel | S |
| `is_canceled` | 1 if the booking was canceled (including no-shows): **target** | – |
| `lead_time` | Days between booking and arrival | S |
| `arrival_date` | Arrival date (built from the raw year, month and day columns) | S (as month) |
| `stays_in_weekend_nights`, `stays_in_week_nights`, `total_nights` | Nights booked | S candidates |
| `adults`, `children`, `babies` | Party size | S (`kids` = any children or babies) |
| `meal` | BB (bed & breakfast), HB (half board), FB (full board), SC (no meal; raw `Undefined` merged in) | S |
| `country` | Country of origin (ISO 3); missing filled with `UNK` | – (doubtful availability) |
| `market_segment` | Online TA, Offline TA/TO, Groups, Direct, Corporate, Complementary, Aviation | S |
| `distribution_channel` | TA/TO, Direct, Corporate, GDS | S |
| `is_repeated_guest` | 1 if the guest stayed before | S candidate |
| `previous_cancellations` | Earlier bookings by the guest that were canceled | S candidate (`prev_cancel`) |
| `previous_bookings_not_canceled` | Earlier bookings by the guest that were not canceled | S (`prev_stay`) |
| `reserved_room_type` | Room type booked (letter code) | S candidate |
| `assigned_room_type` | Room type assigned | – (leakage) |
| `booking_changes` | Number of changes until arrival or cancellation | E |
| `deposit_type` | No Deposit, Non Refund, Refundable | Defines the Non Refund segment |
| `agent` | ID of the travel agency (empty = none) | S (top-10 `agent_group`) |
| `company` | ID of the paying company (empty = none) | S candidate (`has_company_id`) |
| `days_in_waiting_list` | Days on the waiting list before confirmation | E |
| `customer_type` | Transient, Transient-Party, Contract, Group | S |
| `adr` | Average daily rate (room revenue per night; currency not stated in the source) | S candidate |
| `required_car_parking_spaces` | Parking spaces requested | – (leakage) |
| `total_of_special_requests` | Number of special requests | E |
| `reservation_status` | Check-Out, Canceled, No-Show | – (outcome) |
| `reservation_status_date` | Date of the last status: cancellation date, or check-out date | – (outcome; used only to verify timing) |

## Derived columns (`src/features.py`)

| Column | Definition |
|---|---|
| `booking_date` | `arrival_date - lead_time` days (a reconstruction, not a time stamp) |
| `month` | Arrival month |
| `prev_cancel`, `prev_stay` | `previous_cancellations > 0`, `previous_bookings_not_canceled > 0` |
| `kids` | `children + babies > 0` |
| `weekend_share` | Weekend nights / total nights |
| `has_company_id` | `company` is not empty |
| `is_non_refund` | `deposit_type == 'Non Refund'` |
| `booking_value` | `adr × total_nights`: room revenue, **not profit** |
| `agent_group` | Top-10 training agents, `Other`, or `None` |
