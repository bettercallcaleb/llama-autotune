# Changelog

All notable public changes to this project are documented here.

## 0.1.8 — 2026-09-11

First public source release.

Highlights:
- Linux/NVIDIA CUDA environment profiling with explicit CPU fallback.
- llama.cpp build planning and capability discovery from the installed binaries.
- GGUF model inspection and reproducible model/hardware/binary fingerprints.
- Bounded full-offload candidate search across KV cache types and GPU placement strategies.
- `quick` max-context frontier based on full context startup plus a bounded 2K prompt smoke test.
- `normal` and `thorough` modes for stronger validation.
- Per-trial and whole-tune wall-clock guards.
- Persistent result/cache manifests and exact command/report evidence.

Known limits are documented in `README.md` and `AUDIT.md`.
