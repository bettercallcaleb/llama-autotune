# Contributing

Contributions are welcome, especially small reproducible fixes backed by tests or hardware evidence.

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python -m compileall -q src tests
```

## What makes a useful issue

For tuning or hardware bugs, include:
- OS and architecture;
- GPU model(s), VRAM, driver, and toolkit version;
- llama.cpp commit/build string;
- exact `llama-autotune` command;
- `--verbose` output or a sanitized report;
- expected versus observed behavior.

Do **not** publish model files, API keys, private paths, hostnames, GPU UUIDs, corporate data, or other machine-specific identifiers unless they are intentionally sanitized.

## Pull requests

Keep changes narrow. Add or update tests for policy, parsing, process cleanup, or frontier behavior. Avoid hard-coded model paths or hardware IDs. Hardware-specific experiments should be opt-in and driven by environment variables.

The project deliberately distinguishes observation from proof. For example, multi-device VRAM allocation is evidence that multiple devices were active; it is not, by itself, proof of exact tensor placement.
