# Performance records

One file per release, written by `localflow bench release` (run from `engine/`, with LocalFlow
quit - it needs an engine of its own). Each measures, on the machine it names:

* start-up: seconds until speech, and clean-up, are ready in a fresh engine
* latency and accuracy: the own-voice set (30 takes) streamed at real-time pace - key-up to
  text at p50 and p95, and word error rate
* memory: peak private memory of the engine and of the clean-up server, and the graphics memory
  LocalFlow used
* clean-up quality: the clean-up set - exact matches, must-not violations, model latency, and
  command mode

The command compares the new record with the newest earlier one and names every measure that
got worse by more than its tolerance (`RULES` in `engine/src/localflow/perfrecord.py`). Records
from different machines are not comparable, and the comparison says so.
