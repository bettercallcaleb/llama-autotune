from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Iterator
import json
import math
import socket
import statistics
import threading
import time
import urllib.error
import urllib.request
import uuid
from ..config import AutotuneError, RuntimeConfig, Workload
from ..executor import Executor, Process
from ..llama.capabilities import Capabilities, runtime_args
from ..probe.environment import nvidia_memory
from ..llama.devices import DeviceMapping
from ..tuning.depth import depth_prompt, resolve_depth


class TrialError(AutotuneError):
    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status
        self.details: dict = {}


def failure_kind(text: str) -> str:
    lower = text.lower()
    if any(
        x in lower
        for x in (
            "out of memory",
            "cudaerrormemoryallocation",
            "failed to allocate",
            "std::bad_alloc",
        )
    ):
        return "oom"
    if any(
        x in lower
        for x in ("unsupported", "not supported", "invalid argument", "invalid value")
    ):
        return "unsupported"
    return "startup_failure"


def available_port(port: int = 0) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", port))
        return sock.getsockname()[1]


class Server:
    def __init__(self, child: Process, port: int, key: str | None, stop):
        self.child = child
        self.url = f"http://127.0.0.1:{port}"
        self.key = key
        self.stop = stop
        self.properties: dict = {}

    def request(
        self, endpoint: str, payload: dict | None = None, timeout: float = 10
    ) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            self.url + endpoint, data=data, headers=headers
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(request, timeout=timeout) as response:
                value = json.loads(response.read(16_000_001))
        except urllib.error.HTTPError as exc:
            if payload is None:
                raise
            detail = exc.read(16384).decode("utf-8", "replace")
            raise TrialError("request_error", f"HTTP {exc.code}: {detail}") from exc
        if not isinstance(value, dict):
            raise TrialError("malformed_output", "Expected JSON object from server")
        return value

    def ready(self, config: RuntimeConfig, workload: Workload) -> None:
        deadline = time.monotonic() + workload.startup_timeout
        while time.monotonic() < deadline:
            if self.child.process.poll() is not None:
                out, err = self.child.output()
                raise TrialError(
                    failure_kind(out + err),
                    f"Server exited {self.child.process.returncode}: {err[-2000:]}",
                )
            try:
                health = self.request(
                    "/health", timeout=min(2, max(0.1, deadline - time.monotonic()))
                )
                if health.get("status") == "ok":
                    break
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    try:
                        self.request("/props", timeout=2)
                        break
                    except (OSError, ValueError):
                        pass
                elif exc.code != 503:
                    raise TrialError(
                        "startup_failure", f"Health endpoint returned HTTP {exc.code}"
                    ) from exc
            except (OSError, ValueError):
                pass
            time.sleep(0.15)
        else:
            raise TrialError(
                "startup_timeout",
                f"Server startup timed out; logs: {self.child.stderr_path}",
            )
        if self.child.process.poll() is not None:
            raise TrialError("server_crash", "Server exited during readiness check")
        try:
            self.properties = self.request("/props", timeout=5)
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise TrialError(
                    "startup_failure", f"Properties endpoint returned HTTP {exc.code}"
                ) from exc
        actual_ctx = self.properties.get("default_generation_settings", {}).get("n_ctx")
        actual_slots = self.properties.get("total_slots")
        if actual_ctx is not None and int(actual_ctx) < config.ctx // config.parallel:
            raise TrialError(
                "context_mismatch",
                f"Server reduced per-slot context to {actual_ctx}; requested {config.ctx//config.parallel}",
            )
        if actual_slots is not None and int(actual_slots) != config.parallel:
            raise TrialError(
                "context_mismatch",
                "Server concurrency differs from requested parallelism",
            )


@contextmanager
def launch(
    ex: Executor,
    cap: Capabilities,
    model: str,
    config: RuntimeConfig,
    workload: Workload,
    fitting: bool = False,
    margin: int | str = 512,
    port: int = 0,
) -> Iterator[Server]:
    selected = available_port(port)
    args = runtime_args(cap, model, config, fitting=fitting, margin=margin)
    args += [cap.flag("--host"), "127.0.0.1", cap.flag("--port"), str(selected)]
    key = uuid.uuid4().hex if cap.has("--api-key") else None
    if key:
        args += ["--api-key", key]
    with ex.managed(args) as child:
        server = Server(child, selected, key, lambda: ex.stop(child.process))
        try:
            server.ready(config, workload)
            yield server
        except TrialError as exc:
            out, err = child.output()
            exc.details = {
                "argv": args,
                "stdout": out,
                "stderr": err,
                "log": str(child.stderr_path),
                "exit_status_before_cleanup": child.process.poll(),
            }
            raise


class MemoryMonitor:
    def __init__(
        self,
        ex: Executor,
        enabled: bool,
        workload: Workload,
        selected_uuids: list[str] | None = None,
    ):
        self.ex, self.enabled, self.workload = ex, enabled, workload
        self.selected_uuids = (
            set(selected_uuids) if selected_uuids is not None else None
        )
        self.samples: list[dict] = []
        self.lowest: dict[str, dict] = {}
        self.inventory: set[str] | None = None
        self.missing = False
        self.done = threading.Event()
        self.thread: threading.Thread | None = None

    def sample(self) -> None:
        if self.enabled:
            rows = nvidia_memory(self.ex)
            if self.selected_uuids is not None:
                rows = [g for g in rows if g["uuid"] in self.selected_uuids]
                if {g["uuid"] for g in rows} != self.selected_uuids:
                    self.missing = True
            ids = {g["uuid"] for g in rows}
            if rows:
                self.ex.gpu_snapshot = " ".join("{} used={:.0f}MiB free={:.0f}MiB".format(g["uuid"], g["total_mb"]-g["free_mb"], g["free_mb"]) for g in rows)
            if not rows or (self.inventory is not None and ids != self.inventory):
                self.missing = True
            if self.inventory is None:
                self.inventory = ids
            for gpu in rows:
                old = self.lowest.get(gpu["uuid"])
                if old is None or gpu["free_mb"] < old["free_mb"]:
                    self.lowest[gpu["uuid"]] = gpu
            self.samples.append({"time": time.time(), "gpus": rows})
            if len(self.samples) > 4096:
                del self.samples[:2048]

    def __enter__(self):
        if self.enabled:
            self.sample()

            def loop():
                while not self.done.wait(0.5):
                    self.sample()

            self.thread = threading.Thread(target=loop, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *args):
        self.done.set()
        if self.thread:
            self.thread.join(timeout=25)
        return False

    def check(self) -> None:
        if not self.enabled:
            return
        if self.missing or not self.samples:
            raise TrialError(
                "telemetry_unavailable", "Cannot verify NVIDIA memory headroom"
            )
        for gpu in list(self.lowest.values()):
            margin = max(
                self.workload.headroom_mb,
                gpu["total_mb"] * self.workload.headroom_percent / 100,
            )
            if gpu["free_mb"] < margin:
                raise TrialError(
                    "unsafe_headroom",
                    f"{gpu['uuid']} has {gpu['free_mb']:.0f} MiB free; requires {margin:.0f}",
                )


def measure(server: Server, config: RuntimeConfig, workload: Workload, deadline: float | None = None) -> dict:
    plan = resolve_depth(config, workload)

    def request(endpoint: str, payload: dict | None = None, timeout: float = 10) -> dict:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Trial exceeded hard wall-clock timeout")
            timeout = min(timeout, workload.request_timeout, max(0.1, remaining))
        else:
            timeout = min(timeout, workload.request_timeout)
        return server.request(endpoint, payload, timeout=timeout)

    prompt, depth_evidence = depth_prompt(request, workload, plan)

    def one(slot: int) -> dict:
        start = time.monotonic()
        value = request(
            "/completion",
            {
                "prompt": prompt,
                "id_slot": slot,
                "n_predict": workload.tokens,
                "temperature": 0,
                "seed": 42,
                "cache_prompt": False,
                "ignore_eos": True,
                "stream": False,
            },
            timeout=workload.request_timeout,
        )
        timings = value.get("timings", {})
        try:
            pp, tg = float(timings["prompt_per_second"]), float(
                timings["predicted_per_second"]
            )
            generated, processed = int(timings["predicted_n"]), int(timings["prompt_n"])
            if (
                not all(math.isfinite(x) and x > 0 for x in (pp, tg))
                or generated != workload.tokens
                or processed < 1
            ):
                raise ValueError("Incomplete or invalid workload")
            evaluated = int(
                value.get(
                    "tokens_evaluated", processed + int(timings.get("cache_n", 0))
                )
            )
            if not len(prompt) - 1 <= evaluated <= len(prompt) + 1:
                raise ValueError(
                    f"Server evaluated {evaluated} prompt tokens; requested depth {len(prompt)}"
                )
            if "id_slot" in value and int(value["id_slot"]) != slot:
                raise ValueError("Server used a different slot")
            settings = value.get("generation_settings", {})
            if "n_ctx" in settings and int(settings["n_ctx"]) < plan.per_slot_ctx:
                raise ValueError(
                    "Server completion context is smaller than requested per-slot context"
                )
            if evaluated + generated > plan.per_slot_ctx or value.get("truncated"):
                raise ValueError(
                    "Prompt/generation exceeded the requested per-slot context"
                )
        except (KeyError, TypeError, ValueError) as exc:
            raise TrialError(
                "malformed_output", f"Invalid completion timings/workload: {exc}"
            ) from exc
        return {
            "pp_ts": pp,
            "tg_ts": tg,
            "generated": generated,
            "processed": processed,
            "wall_seconds": time.monotonic() - start,
            "timings": timings,
            "slot": slot,
            "prompt_token_count": evaluated,
            "generation_token_count": generated,
            "effective_validation_depth": evaluated,
            "slot_verified": "id_slot" in value,
        }

    request(
        "/completion",
        {"prompt": "Hello", "n_predict": 1, "temperature": 0, "cache_prompt": False},
        timeout=workload.request_timeout,
    )
    measurements, rates = [], []
    for _ in range(workload.repeats):
        start = time.monotonic()
        pool = ThreadPoolExecutor(max_workers=config.parallel)
        try:
            rows = list(pool.map(one, range(config.parallel)))
        except BaseException:
            server.stop()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        wall = time.monotonic() - start
        measurements.extend(rows)
        rates.append(sum(r["generated"] for r in rows) / wall)
    if server.child.process.poll() is not None:
        raise TrialError("server_crash", "Server exited during requests")
    return {
        "pp_ts": statistics.median(r["pp_ts"] for r in measurements),
        "tg_ts": statistics.median(r["tg_ts"] for r in measurements),
        "score": statistics.median(r["tg_ts"] for r in measurements),
        "end_to_end_aggregate_ts": statistics.median(rates),
        "requests": measurements,
        "depth": depth_evidence,
        "objective": "median per-slot decode tokens/s at representative context depth",
    }


def trial(
    ex: Executor,
    cap: Capabilities,
    model: str,
    config: RuntimeConfig,
    workload: Workload,
    fitting: bool = False,
    margin: int | str = 512,
    benchmark: bool = False,
    device_mapping: DeviceMapping | None = None,
) -> dict:
    result = {"status": "interrupted", "started": time.time()}
    started_mono = time.monotonic()
    deadline = started_mono + workload.trial_timeout if workload.trial_timeout > 0 else None
    monitor = MemoryMonitor(
        ex,
        config.devices != "none",
        workload,
        device_mapping.monitor_uuids if device_mapping else None,
    )
    if device_mapping:
        result["device_mapping"] = device_mapping.json()
    server = None
    try:
        with monitor:
            effective = workload
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Trial exceeded hard wall-clock timeout before startup")
                effective = replace(workload, startup_timeout=min(workload.startup_timeout, max(0.1, remaining - 5.0)))
            with launch(ex, cap, model, config, effective, fitting, margin) as server:
                result["initialized"] = True
                monitor.sample()
                monitor.check()
                if benchmark:
                    result.update(measure(server, config, effective, deadline=deadline))
                monitor.sample()
                monitor.check()
                out, err = server.child.output()
                result.update(
                    {
                        "status": "ok",
                        "stdout": out,
                        "stderr": err,
                        "argv": server.child.argv,
                        "log": str(server.child.stderr_path),
                        "effective_properties": server.properties,
                    }
                )
    except TrialError as exc:
        result.update(status=exc.status, error=str(exc), **exc.details)
    except (TimeoutError, socket.timeout) as exc:
        result.update(status="trial_timeout", error=str(exc))
    except (OSError, ValueError, AutotuneError) as exc:
        result.update(status="backend_error", error=str(exc))
    finally:
        result["finished"] = time.time()
        result["trial_wall_seconds"] = time.monotonic() - started_mono
        result["memory_samples"] = monitor.samples
        result["minimum_free_memory"] = monitor.lowest
        baseline = {gpu["uuid"]: gpu for sample in monitor.samples[:1] for gpu in sample.get("gpus", [])}
        deltas = {u: max(0.0, float(baseline.get(u, g)["free_mb"]) - float(g["free_mb"])) for u, g in monitor.lowest.items()}
        peaks = {u: float(g["total_mb"]) - float(g["free_mb"]) for u, g in monitor.lowest.items()}
        intended = device_mapping.selected_ids if device_mapping else []
        uuid_by_id = {e["llama_device"]: e.get("uuid") for e in (device_mapping.entries if device_mapping else [])}
        inactive = [d for d in intended if uuid_by_id.get(d) and deltas.get(uuid_by_id[d], 0) < 64]
        result["placement_evidence"] = {
            "selected_devices": intended, "intended_placement": config.devices,
            "tensor_split": config.tensor_split, "split_mode": config.split_mode,
            "baseline_vram_mb": {u: float(g["total_mb"])-float(g["free_mb"]) for u, g in baseline.items()},
            "peak_vram_mb": peaks, "memory_delta_mb": deltas,
            "minimum_free_vram_mb": {u: float(g["free_mb"]) for u, g in monitor.lowest.items()},
            "mapping_confidence": "reliable" if device_mapping and device_mapping.reliable else "ambiguous",
            "inactive_intended_devices": inactive,
            "device_allocation_observed": not inactive and bool(intended),
            "model_placement_proven": False,
        }
        if server:
            result["exit_status"] = server.child.process.returncode
            result["log"] = str(server.child.stderr_path)
    return result
