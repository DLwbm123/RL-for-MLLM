# RSNA pilot v1 startup snapshot — 2026-10-05

Status at the single startup check: **running, audit_B0**. The detached supervisor survived the SSH session (PPID 1); all supervisor and child argv were neutral Python stdin entries. GPU 0 showed the B0 worker with approximately 16,154 MiB allocated while loading its checkpoint. No immediate failure was present. This is a launch confirmation, not a completed experiment result. No recurring monitor was created.

Executed source commit: `e5dd9f2`; branch `codex/r812`. Frozen configuration: `configs/rsna_pilot.json`. Data and limitations: `reports/rsna_protocol.md` and `reports/rsna_data_audit.json`.

Before B0, the independent RSNA smoke passed on 16 real training images: class labels no/yes are both one token, PEFT save/load preserved scores, evidence/stability gradients were nonzero, zero-gate gradients were zero, generation/teacher-forcing maximum logprob difference 0.0008447613, peak allocated memory 18.35 GiB. CPU checks: 12 passed. See `rsna_engineering_checks.json` for aggregate diagnostics.

Authorized chain: B0 → 32-case short-fit → one-epoch B1 SFT → B1 audit. Each stage stops on failure; B2–B6 remain disabled. Test images were not decoded or evaluated. Model smoke is engineering evidence only, not an efficacy or clinical-safety result.

Private data, patient identifiers, images, individual model outputs, weights, raw logs, and deployment paths are excluded from this public snapshot. Runtime PID and locations are retained in the local private launch receipt and remote run directory.
