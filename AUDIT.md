# v0.1.8 engineering audit

Public policy identifier: `v0.1.8-smoke-frontier-1`.

## Scope

v0.1.8 freezes the first public alpha after repeated physical testing on a heterogeneous NVIDIA CUDA machine. It is intended as a bounded systems tuner, not an exhaustive optimizer.

## Key correction in v0.1.8

Earlier quick max-context prototypes conflated two different questions:

1. can a server allocate/start at context N and serve a real request?;
2. can it prefill almost the entire context N inside a short autotuning budget?

Those are not equivalent.

For `quick + max-context`, v0.1.8 now launches the full candidate context and validates it with a bounded prompt (2048 tokens by default) plus decode. Exponential growth and midpoint refinement therefore test practical startup/request viability without forcing near-full prefill at every frontier point.

`normal` and `thorough` remain available when stronger near-full-depth validation is desired.

## Result semantics

Quick frontier output explicitly distinguishes the validation method:

- `frontier_validation_mode=startup_plus_smoke`
- `frontier_probe_prompt_tokens`
- `max_smoke_validated_ctx`

A timeout is not automatically reported as an OOM or unsafe context. Allocation evidence and completed-request evidence are reported separately where possible.

## Candidate search

The first meaningful candidates use full model offload. Partial NGL is fallback/diagnostic work, not the primary search dimension.

Candidate search includes bounded KV tiers and placement families, including single-device and, where supported/mappable, heterogeneous multi-GPU/tensor-split configurations.

A multi-device command line is never treated as proof of exact model placement. VRAM deltas support the more conservative statement that allocation was observed on the selected devices.

## Cost controls

Default quick guards:

- per trial: 60 seconds;
- whole tune: 600 seconds.

Quick max-context skips redundant coarse/target-depth/server-validation tournament work and normally frontiers a bounded capacity-selected finalist.

## Local validation performed before public packaging

The sanitized public snapshot was checked with:

- focused frontier / cross-version regression tests: 57 passed;
- non-process suite: 129 passed;
- process/server suite independently: 27 passed;
- `python -m compileall -q src`: pass;
- CLI help/import smoke: pass;
- package wheel build: pass;
- privacy scan for previously observed local usernames, paths, GPU UUIDs and temporary tokens: clean.

The project does **not** claim that the monolithic combined pytest run is fully reliable: an older order-dependent process-cleanup hang can still appear when all process tests follow the rest of the suite in one interpreter. The affected subsets pass independently. This is a known engineering limitation rather than hidden CI evidence.

## Public-package sanitization

Historical physical evidence files were intentionally excluded because they contained machine-specific paths and hardware identifiers. The physical CUDA validation helper was converted to environment-variable-driven paths before packaging.

## Known limitations

- Quick smoke validation does not establish long-context model quality.
- Quick smoke validation does not prove that near-full prefill/decode is stable at the reported maximum.
- Search breadth is budget-bounded rather than globally exhaustive.
- Non-NVIDIA and non-Linux backends need more work.
- Placement telemetry is evidence, not exact tensor-placement proof.
- Context values above a model-declared native context, when forced with `--max-context`, are deliberately out-of-spec experiments.
