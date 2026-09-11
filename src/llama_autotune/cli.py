from pathlib import Path
import argparse
import json
import platform
import signal
import sqlite3
import sys
import time
from .build.planner import compile_plan, plan
from .config import AutotuneError, POLICY, RuntimeConfig, Workload, cache_root, digest
from .executor import Executor
from .llama.capabilities import discover, find_binary
from .model.gguf import inspect
from .probe.environment import probe
from .reporting.report import report
from .server.runner import MemoryMonitor, launch
from .storage.store import Store
from .tuning.engine import tune
from .tuning.depth import resolve_depth
from .llama.devices import map_devices


def positive(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return n


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llama-autotune")
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("probe", "build", "inspect", "tune", "run", "report"):
        s = sub.add_parser(command)
        s.add_argument("--cache-dir", type=Path, default=cache_root())
        s.add_argument("--json", action="store_true")
        s.add_argument("--verbose", action="store_true")
        s.add_argument("--debug", action="store_true", help="Print raw subprocess commands and diagnostics")
        if command in ("inspect", "tune", "run", "report"):
            s.add_argument("model", type=Path)
        if command in ("inspect", "tune", "run"):
            s.add_argument("--rehash", action="store_true")
        if command in ("build", "tune", "run"):
            s.add_argument("--build-dir")
        if command in ("tune", "run"):
            s.add_argument("--server", help="Explicit llama-server binary path")
        if command == "build":
            s.add_argument("--source", type=Path, required=True)
            s.add_argument("--backend", choices=("auto", "cuda", "cpu"), default="auto")
            s.add_argument("--plan-only", action="store_true")
            s.add_argument("--jobs", type=positive, default=2)
            s.add_argument("--timeout", type=positive, default=3600)
        if command == "tune":
            s.add_argument("--bench", help="Explicit llama-bench binary path")
            s.add_argument("--ctx", type=positive, required=True, help="Total server context; per-slot context is ctx/parallel")
            s.add_argument("--parallel", type=positive, default=1)
            s.add_argument("--depth", default="auto", help="auto or target prompt token depth per slot")
            s.add_argument("--depth-ratio", type=float, default=0.75, help="Fraction of per-slot context used by auto depth")
            s.add_argument("--batch", type=positive, default=512)
            s.add_argument("--ubatch", type=positive, default=128)
            s.add_argument("--backend", choices=("auto", "cuda", "cpu"), default="auto")
            s.add_argument("--split-mode")
            s.add_argument("--tensor-split")
            s.add_argument("--devices")
            s.add_argument("--vram-headroom-mb", type=int, default=512)
            s.add_argument("--vram-headroom-percent", type=float, default=5)
            s.add_argument("--startup-timeout", type=positive, default=180)
            s.add_argument("--request-timeout", type=positive, default=300)
            s.add_argument("--trial-timeout", type=positive, default=60, help="Hard wall-clock cap in seconds for one tuning/benchmark trial")
            s.add_argument("--tune-timeout", type=positive, default=600, help="Hard wall-clock budget in seconds for the whole tune command")
            s.add_argument("--tokens", type=positive, default=64)
            s.add_argument("--repeats", type=positive, default=2)
            s.add_argument("--finalists", type=positive, default=3)
            s.add_argument("--prompt-file", type=Path)
            s.add_argument("--retry-failed", action="store_true")
            s.add_argument("--context-frontier", choices=("auto", "off", "full"), default="auto", help="auto is budget-aware; off validates target only; full explicitly discovers frontiers")
            s.add_argument("--max-context", type=positive)
            s.add_argument("--context-resolution", type=positive, default=4096)
            s.add_argument("--context-headroom-tokens", type=int, default=8192)
            s.add_argument("--context-headroom-percent", type=float, default=5)
            s.add_argument("--frontier-reserve-tokens", type=positive, default=256)
            s.add_argument("--frontier-smoke-tokens", type=positive, default=2048, help="Quick max-context frontier prompt depth; server still starts at the full candidate context")
            s.add_argument("--budget", choices=("quick", "normal", "thorough"), default="normal")
            s.add_argument("--objective", choices=("performance", "max-context", "balanced"), default="performance")
            s.add_argument("--explore-partial-offload", action="store_true", help="Run diagnostic partial-NGL exploration even when full offload works")
        if command == "run":
            s.add_argument("--port", type=int, default=8080)
        if command == "report":
            s.add_argument("--scope")
    return p


def output(value: dict) -> None:
    print(json.dumps(value, indent=2, sort_keys=True), flush=True)


def execute(args) -> dict:
    store = Store(args.cache_dir)
    ex = Executor(store.root / "logs", args.verbose, args.debug)
    if args.command == "report":
        return report(store, args.model, args.scope)
    if args.command == "inspect":
        return inspect(args.model, store, args.rehash)
    env = probe(ex)
    store.write("environment", env["fingerprint"], env)
    if args.command == "probe":
        return env
    if platform.system() != "Linux":
        raise AutotuneError("v0 build/tune/run supports Linux (including Linux WSL/containers) only")
    with store.lock():
        if args.command == "build":
            directory = Path(args.build_dir) if args.build_dir else store.root / "builds" / (env["fingerprint"][:16] + "-" + args.backend)
            proposal = plan(env, args.source, directory, args.backend)
            return proposal.json() if args.plan_only else compile_plan(proposal, env, ex, store, args.jobs, args.timeout)
        model = inspect(args.model, store, args.rehash)
        profile = store.read("profiles", digest(model["path"]) + "-latest") if args.command == "run" else None
        build = store.read("builds", "latest")
        build_dir, explicit = args.build_dir, args.server
        if not explicit and not build_dir:
            if profile:
                explicit = profile["server_path"]
            elif build:
                build_dir = build["plan"]["directory"]
        binary = find_binary("llama-server", explicit, build_dir)
        if not binary:
            raise AutotuneError("llama-server not found; supply --server /path/to/llama-server or --build-dir /path/to/build, or run build")
        cap = discover(binary, ex)
        if build and build.get("server_fingerprint") != cap.fingerprint:
            build = None
        if args.command == "tune":
            if args.ubatch > args.batch or args.ctx // args.parallel <= args.tokens + 8:
                raise AutotuneError("Require ubatch <= batch and ctx/parallel > tokens+8")
            if args.vram_headroom_mb < 0 or not 0 <= args.vram_headroom_percent < 100:
                raise AutotuneError("Invalid VRAM headroom")
            bench_path = find_binary("llama-bench", args.bench, build_dir or str(binary.parent))
            bench = None
            if bench_path:
                try:
                    bench = discover(bench_path, ex)
                except AutotuneError as exc:
                    print(f"Benchmark discovery failed; using server: {exc}", file=sys.stderr)
            base = RuntimeConfig(999, ctx=args.ctx, parallel=args.parallel, batch=args.batch, ubatch=args.ubatch, split_mode=args.split_mode, tensor_split=args.tensor_split, devices=args.devices)
            workload = Workload(tokens=args.tokens, depth=args.depth, depth_ratio=args.depth_ratio, repeats=args.repeats, startup_timeout=args.startup_timeout, request_timeout=args.request_timeout, trial_timeout=args.trial_timeout, tune_timeout=args.tune_timeout, headroom_mb=args.vram_headroom_mb, headroom_percent=args.vram_headroom_percent, context_frontier=args.context_frontier, max_context=args.max_context, context_resolution=args.context_resolution, context_headroom_tokens=args.context_headroom_tokens, context_headroom_percent=args.context_headroom_percent, frontier_reserve_tokens=args.frontier_reserve_tokens, frontier_smoke_tokens=args.frontier_smoke_tokens, budget=args.budget, objective=args.objective, explore_partial_offload=args.explore_partial_offload, prompt=(args.prompt_file.read_text() if args.prompt_file else Workload().prompt))
            resolve_depth(base, workload)
            if workload.context_headroom_tokens < 0:
                raise AutotuneError("Context headroom tokens cannot be negative")
            return tune(store, ex, env, model, cap, bench, base, workload, args.backend, args.finalists, args.retry_failed, build)
        if not profile:
            raise AutotuneError("No selected profile. Run tune first.")
        if model["fingerprint"] != profile["model_fingerprint"] or env["fingerprint"] != profile["hardware_fingerprint"] or cap.fingerprint != profile["server_fingerprint"]:
            raise AutotuneError("Profile is stale: model, hardware/software, or binary changed. Run tune again.")
        if not 0 <= args.port <= 65535:
            raise AutotuneError("Port must be between 0 and 65535")
        config, workload = RuntimeConfig(**profile["config"]), Workload(**profile["workload"])
        if profile.get("identity", {}).get("policy") != POLICY:
            raise AutotuneError("Profile search policy is stale; run tune again")
        mapping = map_devices(cap.devices, env["gpus"], config.devices, workload, env["visibility"])
        record = {"started": time.time(), "status": "interrupted"}
        server = None
        try:
            with MemoryMonitor(ex, config.devices != "none", workload, mapping.monitor_uuids) as memory:
                with launch(ex, cap, model["path"], config, workload, port=args.port) as server:
                    memory.sample(); memory.check()
                    output({"status": "running", "url": server.url, "api_key": server.key, "config": config.json(), "device_mapping": mapping.json(), "pid": server.child.process.pid, "log": str(server.child.stderr_path)})
                    while server.child.process.poll() is None:
                        memory.check(); time.sleep(0.5)
                    record.update(status="exited", exit_status=server.child.process.returncode)
                    if server.child.process.returncode:
                        raise AutotuneError(f"Server exited {server.child.process.returncode}")
        except AutotuneError as exc:
            record.update(status="failed", error=str(exc)); raise
        finally:
            record["finished"] = time.time()
            if server:
                record["exit_status"] = server.child.process.returncode
            store.add(profile["scope"], config.json(), "run", record)
        return record


def main() -> int:
    args = parser().parse_args()
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        output(execute(args)); return 0
    except KeyboardInterrupt:
        if args.json: output({"status": "interrupted"})
        else: print("Interrupted; managed processes stopped.", file=sys.stderr)
        return 130
    except (AutotuneError, OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
        if args.json: output({"status": "error", "error": str(exc)})
        else: print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)
