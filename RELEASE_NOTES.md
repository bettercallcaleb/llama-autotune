# v0.1.8 public release notes

`llama-autotune` is an early-stage Python CLI for profiling a llama.cpp environment and finding a bounded, evidence-backed runtime configuration.

The most mature path in this release is Linux + NVIDIA CUDA. The tool discovers capabilities from the actual `llama-server` / `llama-bench` binaries instead of assuming a particular llama.cpp version.

For quick max-context searches, the tuner launches the server with the full candidate context (32K, 64K, 128K, ...), then sends a bounded prompt (2048 tokens by default) plus a small generation request. This tests context allocation/startup and basic usability without turning every frontier probe into a near-full-context prefill benchmark.

This release does **not** claim global optimality. Candidate generation and frontier breadth are deliberately bounded to keep runtime practical. Non-NVIDIA backends, accuracy-aware KV selection, exhaustive multi-GPU placement optimization, and stronger cross-platform support remain future work.
