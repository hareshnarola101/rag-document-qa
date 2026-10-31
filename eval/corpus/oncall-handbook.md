# Platform team on-call handbook

## Rotation

On-call runs in weekly shifts that start every Monday at 10:00 local time. Each shift has a
primary and a secondary engineer. The secondary covers if the primary does not acknowledge a
page within 10 minutes, and takes over completely if the primary is sick. New team members
shadow two full shifts before joining the rotation as secondary.

## Severity levels and response times

SEV1 means a customer-facing outage or data loss. The primary must acknowledge a SEV1 page
within 15 minutes and open an incident channel immediately. SEV2 means degraded service with
a workaround available; acknowledge within 30 minutes. SEV3 covers internal tooling issues
and is handled during business hours the next working day.

## Escalation

If a SEV1 is not mitigated within one hour, the on-call engineer escalates to the engineering
manager on duty. For incidents involving payments, the payments team lead is paged at the
same time, regardless of severity. Nobody is ever blamed for escalating too early.

## Handoff

At the end of a shift the outgoing primary writes a handoff note in the on-call channel. The
note lists open incidents, alerts that fired more than twice, and any risky deploys planned
for the coming week.

## Postmortems

Every SEV1 and SEV2 gets a written postmortem within five working days. Postmortems are
blameless: they describe what happened, why the system allowed it, and the follow-up actions
with owners and due dates.

## Compensation

Engineers receive one day of time off in lieu for each week of primary on-call. Being paged
between 22:00 and 07:00 also earns a half day off, up to two extra days per shift.
