# Recorded-context ordering environment

Private `partner_if` / `pif_029` pairing. This Environment Server owns the
multi-step ordering evaluation; `partner_if` owns the unchanged shared verifier.
It uses Gym's typed episode lifecycle and flat-row compatibility bridge.

For each episode it seeds resources, seeds an agent step, activates it, closes
it with confirmed cleanup, then verifies the prefix. A safe undecided step can
continue from an exact saved replay prefix without resending earlier policy
requests. It stops on a decisive pass/fail, a technical exclusion, a safe final
without the required check, or the declared budget. It never invents an answer.
Resources close on success, cancellation and errors via bounded cleanup.

The default limits are six parent steps, eight total parent/worker calls and a
900-second episode deadline. Each policy call has a 180-second timeout.
Checkpoint mode is `restart_only`; cross-process restoration is not supported.
The benchmark's README documents configuration, evidence and current limits.
