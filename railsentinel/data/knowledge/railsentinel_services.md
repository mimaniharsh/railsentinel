---
title: How RailSentinel services work
source: RailSentinel platform documentation
url: https://github.com/mimaniharsh/railsentinel#readme
---
## PNR status on RailSentinel
RailSentinel shows the status of PNRs stored in its own demo reservation system. A PNR is a 10-digit number. The status is CNF for a confirmed berth, RAC for a shared side-lower berth, or WL followed by a position for a waitlisted ticket. These PNRs are demo records and are not linked to IRCTC.

## Confirmation predictions
For RAC and waitlisted tickets, RailSentinel estimates the probability that the ticket will be confirmed by chart preparation. The estimate comes from a model trained on the platform's railway simulation and is shown with its confidence and the main factors behind it. It is an estimate, not a guarantee.

## Booking through the travel planner
The travel planner agent compares itineraries and proposes one, but it never books on its own. A booking is created only after the passenger presses Approve, and it creates a demo PNR in RailSentinel's own reservation system.

## Refund tracking on RailSentinel
When a RailSentinel demo booking is cancelled, the refund is calculated with the published Indian Railways cancellation rules and moves through the stages initiated, processing and credited to original payment method.

## Delay predictions and alerts
RailSentinel predicts arrival delays with an 80% range. If a booked train is likely to arrive more than 60 minutes late, or a waitlisted ticket has a low chance of confirmation, RailSentinel raises an alert and suggests alternative trains.

## Explanation passports
Every prediction comes with a signed explanation passport that records the model version, the confidence, the main factors, the data-quality check and the model's governance status. The signature lets anyone check that the passport has not been altered.

## Data sources
Station names and locations come from the open datameet Indian Railways dataset. Trains, fares, availability and delays in RailSentinel are simulated by its digital twin and are not real Indian Railways data.
