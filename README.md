# llama-autotune

**Early alpha — v0.1.8.** A bounded, evidence-driven environment profiler and runtime autotuner for [llama.cpp](https://github.com/ggml-org/llama.cpp).

`llama-autotune` profiles the local machine and llama.cpp binaries, inspects GGUF models, explores full-offload KV/cache and GPU-placement candidates, validates them with real server requests, and records reproducible evidence about what actually worked.

The most mature path today is **Linux + NVIDIA CUDA**. The project does **not** claim globally optimal tuning.

## Why this exists

llama.cpp exposes many interacting runtime choices: GPU layer offload, KV cache types, flash attention, device placement, tensor split, context size, batch sizes, and more. The fastest or largest-context configuration is hardware- and model-dependent, especially on heterogeneous multi-GPU machines.

This project tries to answer those questions with bounded measurement rather than static guesses.

## v0.1.8 quick max-context semantics

For:

```bash
--budget quick --objective max-context
```

frontier validation is deliberately lightweight:

1. Start `llama-server` with the **full candidate context** (`32K`, `64K`, `128K`, ...).
2. Require real model/KV/cache/graph allocation to succeed.
3. Send a bounded prompt — **2048 tokens by default** — plus a small decode request.
4. If startup and the request complete without OOM/crash/invalid configuration, that context point passes.
5. Grow exponentially until failure, then refine the bracket by midpoint search.

A 128K probe therefore does **not** process a 128K prompt. It tests whether a real 128K server can start and serve a non-trivial request. Stronger near-full-depth validation remains available in `normal` / `thorough` modes.

The report explicitly records:

```text
frontier_validation_mode = startup_plus_smoke
frontier_probe_prompt_tokens = 2048
max_smoke_validated_ctx = ...
```

so this result is not confused with a full-context stress benchmark.

## Install

Python 3.11+ is required.

```bash
git clone https://github.com/bettercallcaleb/llama-autotune.git
cd llama-autotune
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
llama-autotune --help
```

To build llama.cpp through the tool you also need Git, CMake, a C/C++ toolchain, and for CUDA builds a compatible NVIDIA driver/toolkit including `nvcc`.

## Typical workflow

Probe the environment:

```bash
llama-autotune probe --json
```

Inspect a model:

```bash
llama-autotune inspect /path/to/model.gguf --json
```

Plan/build llama.cpp:

```bash
llama-autotune build \
  --source "$HOME/llama.cpp" \
  --build-dir "$HOME/llama-build" \
  --plan-only

llama-autotune build \
  --source "$HOME/llama.cpp" \
  --build-dir "$HOME/llama-build" \
  --backend auto \
  --jobs 4
```

Find a practical max-context configuration:

```bash
llama-autotune tune /path/to/model.gguf \
  --build-dir "$HOME/llama-build" \
  --ctx 32768 \
  --parallel 1 \
  --batch 2048 \
  --ubatch 512 \
  --budget quick \
  --objective max-context \
  --context-frontier full \
  --context-resolution 8192 \
  --context-headroom-percent 5 \
  --verbose
```

Override the model-declared context ceiling for deliberate out-of-spec exploration:

```bash
llama-autotune tune model.gguf \
  --build-dir "$HOME/llama-build" \
  --ctx 32768 \
  --objective max-context \
  --budget quick \
  --context-frontier full \
  --max-context 1048576 \
  --frontier-smoke-tokens 2048
```

Tune for requested-context performance instead:

```bash
llama-autotune tune model.gguf \
  --build-dir "$HOME/llama-build" \
  --ctx 32768 \
  --budget normal \
  --objective performance \
  --context-frontier auto
```

Run the selected profile:

```bash
llama-autotune run model.gguf --port 8080
```

Inspect the stored report:

```bash
llama-autotune report model.gguf --json
```

## What it measures

The current implementation can collect and reason about:

- OS/kernel/CPU/RAM/cgroup information;
- NVIDIA GPU model, VRAM, UUID/PCI mapping, driver and compute capability;
- CUDA/compiler/build-tool availability;
- actual `llama-server --help`, `--version`, and `--list-devices` capabilities;
- GGUF architecture, block count, tensor-type histogram and model fingerprint;
- full-offload feasibility;
- f16/q8/q4 KV cache candidates when supported;
- single-GPU, layer-auto multi-GPU, and bounded tensor-split candidates;
- prompt/decode throughput;
- context startup/smoke or near-full frontier behavior;
- per-device VRAM telemetry and conservative placement evidence;
- failures, timeouts and exact commands/log paths.

The tool deliberately distinguishes **observed multi-device allocation** from proof of exact tensor placement.

## Search policy

The tuner is intentionally bounded. It does not enumerate the Cartesian product of every possible flag.

The general funnel is:

```text
capability discovery
→ full-offload anchors
→ startup + functional smoke
→ bounded pruning
→ objective-specific measurement
→ real server validation
→ optional context frontier
```

For `quick + max-context`, redundant performance tournament stages are skipped. Capacity-efficient q4/q8 and viable multi-GPU placements are retained, then the frontier itself is authoritative.

Defaults include a **60-second per-trial hard cap** and a **600-second whole-tune wall-clock budget**. Both are configurable.

## Safety and reproducibility

- External commands use argument arrays rather than a shell.
- The server binds to `127.0.0.1`.
- Managed POSIX process groups are terminated and reaped on normal cleanup/interruption.
- Hidden `LLAMA_ARG_*` overrides are stripped from child environments.
- Profiles fingerprint hardware, model, binary capabilities and search policy.
- Successful and failed evidence is cached under `~/.cache/llama-autotune` (or `XDG_CACHE_HOME`).
- Local reports may contain paths, hardware identifiers and model names; sanitize them before posting publicly.

No model, prompt, hardware inventory, or tuning result is intentionally sent to a remote service by this project.

## Known limitations

- Linux/NVIDIA CUDA is the best-tested path in v0.1.8.
- Windows/macOS and non-NVIDIA accelerator support are incomplete.
- Quick max-context smoke proves startup + bounded request viability, **not** near-full-context quality or performance.
- Candidate search is bounded, so the selected configuration is not guaranteed to be globally optimal.
- KV quantization selection is not accuracy-aware yet.
- VRAM telemetry is sampled and cannot prove every instantaneous peak.
- Exact tensor placement is not inferred solely from command-line intent or VRAM deltas.
- A known order-dependent process-cleanup race can make the monolithic local pytest suite hang on some runs; the process/server and non-process suites pass independently. See `AUDIT.md`.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m compileall -q src
```

Hardware-specific validation should use sanitized environment-variable-driven paths rather than committing local model paths or GPU identifiers.

## License

MIT. Not affiliated with llama.cpp.
