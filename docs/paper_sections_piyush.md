# ICARUS paper draft: data, TLE cleaning, and precise-orbit validation

_Draft sections for integration into the team manuscript. Values below describe the
current real-data run; the causal and operator results should be updated if the team
freezes a later analysis table._

## Data and cohort construction

We study orbital decay during geomagnetic storms using publicly available two-line
element (TLE) records and hourly space-weather data. The primary observation window is
April 2024 through April 2025. We define storms from the storm catalogue used by the
analysis pipeline and retain 20 events with minimum Dst below −50 nT. The May 2024
event is the strongest event in the window, with a minimum Dst of approximately −406 nT.

The main sample contains 250 passive objects between 400 and 800 km altitude. We selected
objects with coverage on at least eight of nine Space-Track snapshot dates and allocated
the sample across 100-km altitude bands rather than selecting by NORAD number. Debris and
rocket bodies form the passive sample because they cannot actively correct their orbit.
We additionally retain 15 objects at 1,000–1,500 km as a high-altitude negative-control
group. The processed passive table contains 265 objects and 199,580 adjacent TLE
intervals in the first complete real-data build; the later cleaned download contains
203,954 deduplicated TLE states and 198,388 usable intervals after the revised checks.

Active payloads are analysed separately for the operator-offset question. The active
cohort contains 25 Starlink satellites and 25 other payloads selected using the same
coverage principle. The active-satellite pipeline preserves station-keeping intervals
and does not apply passive-object climb removal. The final joined object catalogue
contains 291 objects. We do not include raw TLE or processed parquet files in version
control; the repository stores the code and manifests needed to reproduce the tables.

## TLE cleaning and decay intervals

We parse each TLE with SGP4 and convert the Brouwer mean motion to a mean semi-major axis
using the WGS-72 gravitational constant. Each record retains the epoch, eccentricity,
inclination, right ascension of the ascending node, local solar time, TLE lines, and
publication timestamp. We first collapse re-issued records with effectively identical
epochs, keeping the most recently published record. This prevents duplicate publication
records from creating artificial zero-length or very short decay intervals.

For each object, adjacent cleaned states define an interval. We reject states outside
the eccentricity and altitude bounds, cross-tag discontinuities, and endpoint failures
identified by the TLE consistency checks. We omit intervals spanning more than five days
and intervals that cross a storm onset, because neither interval can be assigned a clean
pre- or post-onset exposure. In the revised passive run, seven long gaps and 5,294
onset-bridging intervals were omitted.

TLE noise can create a small apparent rise in semi-major axis. We therefore estimate
object-specific robust noise from local median absolute deviations and flag a climb only
when the rise exceeds both a minimum floor and approximately three times the estimated
TLE noise. The flagged climb is retained as a diagnostic rather than automatically
discarding the surrounding record. The revised rule flagged 416 passive intervals; a
tighter point-level diagnostic flagged 15 TLE states (0.0074%), which we report as a
sensitivity check. This distinction matters because deleting only upward observations
would mechanically make decay appear faster.

For each retained interval, the outcome is the change in semi-major axis divided by the
elapsed time, expressed in metres per day. Negative values indicate that the object is
falling. The pipeline records storm identifiers, days relative to each storm's
`main_onset`, cleaning flags, altitude band, and object role so that the event-study and
model-error analyses can apply the same exclusions without reprocessing the raw TLEs.

## GRACE-FO and Swarm precise-orbit validation

We validate the TLE-derived decay signal against reduced-dynamic precise-orbit products
for GRACE-FO-1 and Swarm-A. The precise files provide Cartesian position and velocity
states. We compute the osculating semi-major axis from the vis-viva equation, then
average states over complete orbital revolutions before calculating daily decay rates.
For Swarm-A, the revolution period is estimated from successive radius minima rather
than imposed as a fixed nominal period. The estimated period is 93.83 minutes. Duplicate
timestamps and overlapping states at daily-file boundaries are removed before the
revolution averages are formed.

The October 2024 GRACE-FO comparison provides the clearest validation: the TLE and
precise daily decay-rate series peak on the same day, with a zero-lag correlation of
approximately 0.70 and similar storm-day magnitude (about −0.10 versus −0.12 km/day).
This indicates that TLEs capture the timing and most of the size of the storm decay,
with a modest peak underestimate. The earlier May 2024 level comparison gives a centred
correlation of 0.907, centred RMSE of 0.097 km, and a slope difference of 0.0025 km/day
after removing the mean-element versus osculating-element level offset.

After revolution-based averaging and duplicate removal, Swarm-A no longer shows the
alternating-sign daily pattern produced by drifting fixed bins. Its zero-lag decay-rate
correlation is 0.457. We treat Swarm-A as a supporting cross-check rather than combining
it with the GRACE-FO headline estimate, because its product and sampling characteristics
are different. Together, the two checks support using the cleaned TLE interval outcome
for the cohort analysis while making the remaining measurement uncertainty explicit.

## Reproducibility notes

The relevant implementations are `src/orbit.py` for TLE parsing and interval cleaning,
`src/precise_orbit.py` for precise-orbit parsing and rate comparison, and
`src/build_table.py` for assembling the analysis table. The validation plot and numerical
outputs are generated from ignored `data/` inputs so that credentials and large raw
downloads are never committed.
