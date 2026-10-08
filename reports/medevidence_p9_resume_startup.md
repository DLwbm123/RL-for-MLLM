# P9 continuation startup

Status at 2026-10-08T14:25:45.455409+00:00: **running**, with 64/1024 visual feature records atomically persisted in two 32-patient blocks. GPU model forward and neutral controller/worker command lines were confirmed. This startup receipt is not a final result.

Execution source: `c28aca50e6c7f784e3bb2cd446fa7ec3fe12b4f3`. The continuation reuses the original 2400 detector steps, 960 out-of-fold training candidate sets, 64 calibration candidate sets, and the frozen 256-patient development cache. New detector training: 0 steps. Scientific settings and the primary gate are unchanged.

Four CPU test groups passed, including interrupted extraction, tail-block persistence, recovery of only missing records, feature equivalence, complete-cache model-load avoidance, provenance rejection, diagnostic counts and report export. Preparation checked patient membership and cache readability; no file hashes were computed.

The independent continuation cap is 2700 GPU seconds (45 minutes), with earlier charged work retained at 5855.87869143486 GPU seconds. The cumulative 10800-second cap remains unchanged. The worker lifecycle includes model loading, feature extraction, CPU head fitting, calibration, evaluation and saving. No automatic retry or budget extension is authorized.

Head fitting, calibration, final evaluation and scientific success were not yet established at this check. No ongoing monitoring was scheduled. Patient records, feature chunks, detector checkpoints and raw deployment logs remain private. Aggregate final results and budget receipts are due for publication when completion is confirmed.
