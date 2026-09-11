from dataclasses import dataclass, replace
import math
import re
from typing import Callable
from ..config import AutotuneError, RuntimeConfig, Workload

CONTEXT_PATTERNS = (
    r"out of memory",
    r"cudaerrormemoryallocation",
    r"hip.*(?:out of memory|allocation)",
    r"(?:failed|cannot|unable) to allocate",
    r"kv.*alloc",
    r"compute buffer.*alloc",
    r"backend.*alloc",
    r"prefill.*oom",
    r"decode.*oom",
    r"killed",
    r"memory pressure",
)
NON_CONTEXT_PATTERNS = (
    r"invalid (?:cli |command line )?(?:argument|option)",
    r"unknown (?:argument|option)",
    r"unsupported.*(?:cache|kv|flash|architecture)",
    r"malformed.*gguf",
    r"(?:missing|no such).*(?:model|gguf)",
    r"binary.*incompat",
    r"tokeniz",
    r"assert(?:ion)?.*(?:unsupported|type|argument)",
)
REPEATS = {"quick": 1, "normal": 2, "thorough": 3}


def classify_context_failure(result: dict, correlated_high_context: bool = False) -> str:
    if result.get("status") == "ok": return "success"
    text = " ".join(str(result.get(k, "")) for k in ("status", "error", "stdout", "stderr")).lower()
    if result.get("status") in ("unsupported", "malformed_output"): return "non_context"
    if any(re.search(pattern, text) for pattern in NON_CONTEXT_PATTERNS): return "non_context"
    if any(re.search(pattern, text) for pattern in CONTEXT_PATTERNS): return "context"
    if result.get("status") in ("request_timeout", "trial_timeout", "startup_timeout"): return "timeout"
    if result.get("initialized") and result.get("status") in ("oom", "unsafe_headroom"): return "context"
    if result.get("initialized") and correlated_high_context and result.get("status") in ("server_crash", "backend_error"): return "context"
    return "unknown"


def frontier_validation_mode(workload: Workload) -> str:
    if workload.objective == "max-context" and workload.budget == "quick": return "startup_plus_smoke"
    return "near_full"


def frontier_workload(config: RuntimeConfig, workload: Workload) -> Workload:
    per_slot = config.ctx // config.parallel
    if frontier_validation_mode(workload) == "startup_plus_smoke":
        available = per_slot - workload.tokens - 32
        depth = min(workload.frontier_smoke_tokens, available)
        if depth < 1: raise AutotuneError("Context is too small for frontier smoke and generation")
        return replace(workload, depth=str(depth), depth_ratio=workload.depth_ratio)
    reserve = workload.tokens + workload.frontier_reserve_tokens
    depth = per_slot - reserve
    if depth < 1: raise AutotuneError("Context is too small for frontier reserve and generation")
    return replace(workload, depth=str(depth), depth_ratio=workload.depth_ratio)


def recommended_context(maximum: int, workload: Workload) -> int:
    margin = max(workload.context_headroom_tokens, math.ceil(maximum * workload.context_headroom_percent / 100))
    return max(0, ((maximum - margin) // workload.context_resolution) * workload.context_resolution)


def validate_context_policy(workload: Workload) -> None:
    if workload.context_resolution < 1 or workload.context_headroom_tokens < 0: raise AutotuneError("Context resolution/headroom must be valid")
    if workload.trial_timeout <= 0 or workload.tune_timeout <= 0: raise AutotuneError("Trial/tune timeouts must be positive")
    if not 0 <= workload.context_headroom_percent < 100: raise AutotuneError("Context headroom percent must be in [0, 100)")
    if workload.frontier_reserve_tokens < 1 or workload.frontier_smoke_tokens < 1 or workload.budget not in REPEATS: raise AutotuneError("Invalid frontier reserve or budget")
    if workload.context_frontier not in ("auto", "off", "full"): raise AutotuneError("Invalid context frontier mode")
    if workload.objective not in ("performance", "max-context", "balanced"): raise AutotuneError("Invalid tuning objective")
    if workload.objective == "max-context" and workload.context_frontier == "off": raise AutotuneError("max-context objective requires context-frontier auto or full")


@dataclass(frozen=True)
class FrontierResult:
    max_allocatable_ctx: int | None
    max_allocatable_ctx_is_lower_bound: bool
    max_benchmarkable_ctx: int | None
    recommended_safe_ctx: int | None
    unsafe_upper_ctx: int | None
    status: str
    trials: list[dict]
    timeout_upper_ctx: int | None = None
    validation_mode: str | None = None
    probe_prompt_tokens: int | None = None

    def json(self) -> dict:
        failed_allocations = sorted(t["ctx"] for t in self.trials if not t.get("initialized") and t.get("outcome") == "context")
        return {"max_allocatable_ctx": self.max_allocatable_ctx, "max_allocatable_ctx_is_lower_bound": bool(self.max_allocatable_ctx), "max_allocatable_known_good": self.max_allocatable_ctx, "max_allocatable_failed_upper": failed_allocations[0] if failed_allocations else None, "max_benchmarkable_ctx": self.max_benchmarkable_ctx, "recommended_safe_ctx": self.recommended_safe_ctx, "unsafe_upper_ctx": self.unsafe_upper_ctx, "timeout_upper_ctx": self.timeout_upper_ctx, "status": self.status, "frontier_validation_mode": self.validation_mode, "frontier_probe_prompt_tokens": self.probe_prompt_tokens, "max_smoke_validated_ctx": self.max_benchmarkable_ctx if self.validation_mode == "startup_plus_smoke" else None, "trials": self.trials}


def search_context_frontier(base: RuntimeConfig, workload: Workload, upper: int, evaluate: Callable[[RuntimeConfig, Workload], dict]) -> FrontierResult:
    validate_context_policy(workload)
    resolution = workload.context_resolution
    upper = max(base.ctx, (upper // resolution) * resolution)
    seen: dict[int, str] = {}
    trials: list[dict] = []
    max_allocatable = None
    def run(ctx: int, repeats: int = 1) -> str:
        nonlocal max_allocatable
        ctx = max(1, (ctx // resolution) * resolution)
        outcomes = []
        for attempt in range(repeats):
            config = replace(base, ctx=ctx)
            result = evaluate(config, frontier_workload(config, workload))
            if result.get("initialized"): max_allocatable = max(max_allocatable or 0, ctx)
            kind = classify_context_failure(result, correlated_high_context=any(v == "pass" and n < ctx for n, v in seen.items()))
            outcomes.append(kind)
            trials.append({"ctx": ctx, "attempt": attempt + 1, "outcome": kind, "status": result.get("status", "unknown"), "initialized": bool(result.get("initialized")), "depth": result.get("depth"), "error": result.get("error"), "trial_wall_seconds": result.get("trial_wall_seconds")})
            if kind in ("non_context", "unknown"):
                seen[ctx] = kind; return kind
        if all(x == "success" for x in outcomes): state = "pass"
        elif all(x == "timeout" for x in outcomes): state = "timeout"
        elif all(x == "context" for x in outcomes): state = "fail"
        elif any(x == "timeout" for x in outcomes) and not any(x == "context" for x in outcomes): state = "timeout"
        else: state = "unstable"
        seen[ctx] = state; return state
    def result(low: int | None, high: int | None, high_reason: str | None, status: str | None = None):
        unsafe_points = sorted(t["ctx"] for t in trials if t["outcome"] == "context")
        timeout_points = sorted(t["ctx"] for t in trials if t["outcome"] == "timeout")
        practical_status = status or ("time_limited" if high_reason == "timeout" else "upper_reached" if high is None and low == upper else "bounded")
        safe = low if (low and practical_status == "time_limited") else (recommended_context(low, workload) if low else None)
        return FrontierResult(max_allocatable, bool(max_allocatable), low or None, safe, unsafe_points[0] if unsafe_points else None, practical_status, trials, timeout_points[0] if timeout_points else None, frontier_validation_mode(workload), workload.frontier_smoke_tokens if frontier_validation_mode(workload) == "startup_plus_smoke" else None)
    low = (base.ctx // resolution) * resolution
    if low < 1: low = base.ctx
    first = run(low)
    if first != "pass":
        if first == "timeout": return result(None, low, "timeout", "time_limited")
        if first in ("fail", "unstable"): return result(None, low, "fail", first)
        return result(None, None, None, first)
    high = None; high_reason = None; point = low
    while point < upper:
        candidate = min(upper, point * 2)
        candidate = max(point + resolution, (candidate // resolution) * resolution)
        candidate = min(candidate, upper)
        if candidate <= point: break
        state = run(candidate)
        if state == "pass":
            low = point = candidate
            if candidate == upper: break
        elif state in ("fail", "unstable", "timeout"):
            high = candidate; high_reason = "timeout" if state == "timeout" else "fail"; break
        else: return result(low, None, None, state)
    while high is not None and high - low > resolution:
        middle = (((low + high) // 2) // resolution) * resolution
        if middle <= low: middle = low + resolution
        if middle >= high: break
        state = run(middle)
        if state == "pass": low = middle
        elif state in ("fail", "unstable", "timeout"):
            high = middle; high_reason = "timeout" if state == "timeout" else "fail"
        else: return result(low, high, high_reason, state)
    confirms = REPEATS[workload.budget]
    if low and confirms:
        state = run(low, confirms)
        if state != "pass":
            stable = sorted(n for n, value in seen.items() if value == "pass" and n < low)
            low = stable[-1] if stable else 0
            if state == "timeout":
                high_reason = "timeout"; high = high or (low + resolution if low else base.ctx)
    return result(low, high, high_reason)
