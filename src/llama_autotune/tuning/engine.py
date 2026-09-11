from dataclasses import asdict, replace
import time
from ..config import AutotuneError, POLICY, RuntimeConfig, Workload, digest
from ..executor import Executor
from ..llama.capabilities import Capabilities, parse_bench, parse_offload, runtime_args
from ..server.runner import trial, failure_kind
from ..storage.store import Store
from .policy import bounded_layers, variants, budget_policy, max_context_select
from .depth import resolve_depth
from .frontier import search_context_frontier, validate_context_policy
from ..llama.devices import map_devices, parse_devices

TRANSIENT = {"interrupted", "startup_timeout", "request_timeout", "trial_timeout", "telemetry_unavailable", "unsafe_headroom"}


def cached_trial(
    store: Store,
    scope: str,
    config: RuntimeConfig,
    stage: str,
    execute,
    retry_failed: bool,
) -> dict:
    old = store.previous(scope, config.json(), stage)
    if old and old.get("status") not in TRANSIENT and not retry_failed:
        return {**old, "cache_reused": True}
    try:
        result = execute()
    except BaseException as exc:
        store.add(
            scope,
            config.json(),
            stage,
            {"status": "interrupted", "error": type(exc).__name__},
        )
        raise
    store.add(scope, config.json(), stage, result)
    return result


def coarse(
    ex: Executor,
    cap: Capabilities,
    model: str,
    config: RuntimeConfig,
    workload: Workload,
) -> dict:
    try:
        depth = resolve_depth(config, workload)
        if not cap.has("--n-depth", "-d"):
            return {
                "status": "bench_unavailable",
                "error": "Benchmark has no depth capability; use depth-aware server fallback",
                "depth": depth.json(),
            }
        args = runtime_args(cap, model, config, server=False)
        args += [
            cap.flag("--n-prompt", "-p"),
            str(depth.coarse_prompt_tokens),
            cap.flag("--n-depth", "-d"),
            str(depth.effective_validation_depth),
            cap.flag("--n-gen", "-n"),
            str(workload.tokens),
            cap.flag("--repetitions", "-r"),
            str(workload.repeats),
            cap.flag("--output", "-o"),
            "json",
        ]
        r = ex.run(args, timeout=min(workload.request_timeout, workload.trial_timeout))
        if not r.ok:
            return {
                "status": (
                    "request_timeout"
                    if r.timed_out
                    else failure_kind(r.stdout + r.stderr)
                ),
                "error": r.error or r.stderr[-2000:],
                "log": r.log,
                "argv": r.argv,
            }
        measured = parse_bench(r.stdout)
        for row in measured["rows"]:
            if int(row.get("n_depth", -1)) != depth.effective_validation_depth:
                raise AutotuneError("Benchmark output does not confirm requested depth")
        return {
            "status": "ok",
            **measured,
            "argv": args,
            "log": r.log,
            "depth": depth.json(),
            "score": measured["tg_ts"],
            "objective": "decode tokens/s at representative context depth",
        }
    except (AutotuneError, ValueError, TypeError, KeyError) as exc:
        return {"status": "bench_unavailable", "error": str(exc)}


def tune(store, ex, env, model, cap, bench, base, workload, backend, finalists, retry_failed, build=None):
    from .policy import alternate_fa, diverse_partial_seeds, kv_tier, placement_candidates, placement_family, stratified_select
    validate_context_policy(workload)
    plan = budget_policy(workload.budget)
    cuda = backend != "cpu" and bool(env["gpus"]) and "CUDA" in cap.devices.upper()
    if backend == "cuda" and not cuda:
        raise AutotuneError("CUDA requested but no CUDA device advertised by this binary")
    if not cuda and not cap.has("--device", "-dev"):
        raise AutotuneError("CPU mode requires a binary advertising --device none")
    if env["visibility"].get("GGML_CUDA_ENABLE_UNIFIED_MEMORY"):
        raise AutotuneError("Unset GGML_CUDA_ENABLE_UNIFIED_MEMORY for measurable VRAM fitting")
    depth = resolve_depth(base, workload)
    maximum = model.get("layers")
    if not isinstance(maximum, int) or maximum < 1:
        raise AutotuneError("Cannot discover model layer count; block_count metadata is required")
    maximum += 1
    try:
        search = variants(base, cap, cuda, workload.budget, env, workload)
    except TypeError:
        search = variants(base, cap, cuda)
    full = [replace(config, ngl=maximum) for config in search if config.devices != "none"] if cuda else search
    placement_limit = max(plan.placements, 3) if workload.objective == "max-context" else plan.placements
    _, placement_limitations = placement_candidates(base, cap, env, workload, placement_limit) if cuda else ([base], [])
    identity = {"policy": POLICY, "hardware": env["fingerprint"], "model": model["fingerprint"], "server": cap.fingerprint, "bench": bench.fingerprint if bench else None, "base": base.json(), "workload": asdict(workload), "backend": "cuda" if cuda else "cpu"}
    scope = digest(identity)
    accounting = {name: 0 for name in ("generated_candidates", "search_candidates", "startup_attempts", "model_load_attempts", "smoke_attempts", "coarse_attempts", "target_depth_attempts", "validation_attempts", "frontier_attempts", "frontier_probe_points", "frontier_model_load_attempts")}
    trace = []
    trace_by_key = {}
    def ensure_trace(config, reason):
        key = digest(config.json())
        if key not in trace_by_key:
            row = {"candidate_id": f"C{len(trace)+1:03d}", "config": config.json(), "placement_family": placement_family(config), "kv_tier": kv_tier(config), "fa_mode": config.flash, "stage_entered": [], "reason_admitted": reason, "reason_rejected_or_pruned": None, "measured_score": None}
            accounting["generated_candidates"] += 1
            accounting["search_candidates"] += 1
            trace.append(row); trace_by_key[key] = row
        return trace_by_key[key]
    for config in full:
        ensure_trace(config, "budgeted full-offload KV/placement anchor")
    manifest = {"policy": POLICY, "scope": scope, "environment": env, "model": model, "server": cap.json(), "bench": bench.json() if bench else None, "build": build, "base": base.json(), "workload": asdict(workload), "backend": identity["backend"], "depth": depth.json(), "search_plan": asdict(plan), "placement_planning_limitations": placement_limitations, "cost_accounting": accounting, "candidate_trace": trace}
    store.write("profiles", scope + "-manifest", manifest)
    reasoning = {"stages": [], "partial_ngl": "pending", "frontier": "pending"}
    mappings = {}
    def mapping_for(config):
        key = config.devices or ""
        if key not in mappings:
            mappings[key] = map_devices(cap.devices, env["gpus"], config.devices, workload, env["visibility"])
        return mappings[key]
    started = time.monotonic()
    tune_deadline = started + workload.tune_timeout
    def progress(index, total, config, stage):
        ident = f"candidate={ensure_trace(config, 'adaptive expansion')['candidate_id']} placement={placement_family(config)} KV={kv_tier(config)} FA={config.flash}"
        if hasattr(ex, "current_stage"):
            ex.current_stage = f"{ident} stage={stage}"
        getattr(ex, "progress", lambda message: None)(f"[{index}/{total}] {ident} stage={stage} elapsed={time.monotonic()-started:.0f}s heartbeat")
    def score(result):
        return float(result.get("score", result.get("tg_ts", 0)) or 0)
    stage_counter = {"startup": "startup_attempts", "smoke": "smoke_attempts", "coarse": "coarse_attempts", "coarse-server": "coarse_attempts", "target-depth": "target_depth_attempts", "target-depth-server": "target_depth_attempts", "validate": "validation_attempts", "context-frontier": "frontier_attempts", "partial-feasibility": "startup_attempts", "partial-diagnostic": "startup_attempts"}
    def execute_stage(config, stage, execute, model_load=True):
        if time.monotonic() >= tune_deadline:
            raise AutotuneError(f"Autotune wall-clock budget exceeded ({workload.tune_timeout:.0f}s)")
        row = ensure_trace(config, "adaptive expansion")
        row["stage_entered"].append(stage)
        counter = stage_counter.get(stage)
        if counter:
            accounting[counter] += 1
        result = cached_trial(store, scope, config, stage, execute, retry_failed)
        if model_load and not result.get("cache_reused"):
            accounting["model_load_attempts"] += 1
        if result.get("status") == "ok" and score(result):
            row["measured_score"] = score(result)
        elif result.get("status") != "ok":
            row["reason_rejected_or_pruned"] = f"{stage}: {result.get('status', 'unknown')}"
        return result

    smoke_depth = min(plan.smoke_depth, max(1, base.ctx // base.parallel - workload.tokens - 33))
    smoke_work = replace(workload, depth=str(smoke_depth), repeats=1)
    startup_rows = []
    startup_metrics = {}
    pending = list(full)
    for index, config in enumerate(pending, 1):
        progress(index, len(pending), config, "model startup + functional smoke")
        try:
            mapping = mapping_for(config)
        except AutotuneError as exc:
            result = {"status": "unsupported", "error": str(exc)}
            store.add(scope, config.json(), "static-rejection", result)
            ensure_trace(config, "anchor")["reason_rejected_or_pruned"] = f"static: {exc}"
            continue
        result = execute_stage(config, "startup", lambda c=config, m=mapping: trial(ex, cap, model["path"], c, smoke_work, benchmark=True, device_mapping=m))
        offloaded = parse_offload(result.get("stdout", "") + result.get("stderr", ""))
        if result.get("status") == "ok" and (not cuda or offloaded is None or offloaded >= maximum):
            startup_rows.append((config, score(result)))
            startup_metrics[digest(config.json())] = (result.get("pp_ts"), result.get("tg_ts") or result.get("score"))
            continue
        if result.get("status") == "unsupported":
            alternate = alternate_fa(config, cap)
            if alternate and digest(alternate.json()) not in trace_by_key:
                ensure_trace(alternate, f"alternate FA after {config.flash} was unsupported")
                progress(index, len(pending), alternate, "adaptive FA fallback startup + smoke")
                alternate_result = execute_stage(alternate, "startup", lambda c=alternate: trial(ex, cap, model["path"], c, smoke_work, benchmark=True, device_mapping=mapping_for(c)))
                alternate_offload = parse_offload(alternate_result.get("stdout", "") + alternate_result.get("stderr", ""))
                if alternate_result.get("status") == "ok" and (alternate_offload is None or alternate_offload >= maximum):
                    startup_rows.append((alternate, score(alternate_result)))
                    startup_metrics[digest(alternate.json())] = (alternate_result.get("pp_ts"), alternate_result.get("tg_ts") or alternate_result.get("score"))
    startup_limit = plan.startup_survivors
    if workload.objective == "max-context":
        startup_limit = max(startup_limit, min(len(startup_rows), 4 if workload.budget == "quick" else plan.max_context_frontiers + 2))
        startup_rows = max_context_select(startup_rows, startup_limit, config_of=lambda row: row[0], performance_of=lambda row: row[1])
    else:
        startup_rows = stratified_select(startup_rows, startup_limit, config_of=lambda row: row[0], score_of=lambda row: row[1])
    startup_keys = {digest(row[0].json()) for row in startup_rows}
    for config in full:
        row = ensure_trace(config, "anchor")
        if "startup" in row["stage_entered"] and digest(config.json()) not in startup_keys and row["reason_rejected_or_pruned"] is None:
            row["reason_rejected_or_pruned"] = "pruned after startup by stratified budget"
    reasoning["stages"].append({"stage": "startup", "attempted": accounting["startup_attempts"], "survived": len(startup_rows), "placement_families": sorted({placement_family(row[0]) for row in startup_rows})})
    accounting["initial_model_load_attempts"] = accounting["model_load_attempts"]

    if workload.objective == "max-context":
        smoke_limit = max(plan.smoke_survivors, min(len(startup_rows), 4 if workload.budget == "quick" else plan.max_context_frontiers + 2))
        smoke = max_context_select(startup_rows, smoke_limit, config_of=lambda row: row[0], performance_of=lambda row: row[1])
    else:
        smoke = stratified_select(startup_rows, plan.smoke_survivors, config_of=lambda row: row[0], score_of=lambda row: row[1])
    for config, measured in smoke:
        row = ensure_trace(config, "combined startup and functional smoke")
        row["stage_entered"].append("smoke")
        row["measured_score"] = measured
    accounting["smoke_attempts"] = len(startup_rows)
    reasoning["stages"].append({"stage": "smoke", "attempted": accounting["smoke_attempts"], "survived": len(smoke), "depth": smoke_depth, "model_loads_reused_from_startup": True, "placement_families": sorted({placement_family(row[0]) for row in smoke})})

    if workload.objective == "max-context":
        ranked = list(smoke)
        reasoning["stages"].append({
            "stage": "coarse", "attempted": 0, "survived": len(ranked),
            "skipped": "max-context objective: capacity heuristic goes directly to frontier shortlist",
            "placement_families": sorted({placement_family(row[0]) for row in ranked}),
        })
    else:
        coarse_work = replace(workload, depth=str(smoke_depth), repeats=plan.repeats)
        ranked = []
        for index, (config, _) in enumerate(smoke, 1):
            progress(index, len(smoke), config, "coarse benchmark")
            result = execute_stage(config, "coarse", lambda c=config: coarse(ex, bench, model["path"], c, coarse_work) if bench else {"status": "bench_unavailable"}, model_load=bool(bench))
            if result.get("status") not in ("ok", "request_timeout", "trial_timeout"):
                result = execute_stage(config, "coarse-server", lambda c=config: trial(ex, cap, model["path"], c, coarse_work, benchmark=True, device_mapping=mapping_for(c)))
            if result.get("status") == "ok":
                ranked.append((config, score(result)))
        ranked = stratified_select(ranked, plan.coarse_survivors, config_of=lambda row: row[0], score_of=lambda row: row[1])
        reasoning["stages"].append({"stage": "coarse", "attempted": accounting["coarse_attempts"], "survived": len(ranked), "placement_families": sorted({placement_family(row[0]) for row in ranked})})
    accounting["model_loads_before_target_depth"] = accounting["model_load_attempts"]

    seeds = diverse_partial_seeds(full, plan.partial_seeds)
    if ranked and cuda and workload.explore_partial_offload:
        diagnostic = []
        for variant in seeds:
            def diagnose(ngl, variant=variant):
                if ngl == maximum: return False
                config = replace(variant, ngl=ngl)
                ensure_trace(config, "explicit partial-offload diagnostic seed")
                result = execute_stage(config, "partial-diagnostic", lambda: trial(ex, cap, model["path"], config, smoke_work, benchmark=True, device_mapping=mapping_for(config)))
                if result.get("status") == "ok": diagnostic.append({"config": config.json(), "score": score(result)}); return True
                return False
            bounded_layers(maximum, maximum - 1, diagnose, 4 if workload.budget == "quick" else 6)
        reasoning["partial_diagnostics"] = diagnostic
        reasoning["partial_ngl"] = "explicit diagnostic exploration performed; full-offload candidates retain selection priority"
    if not ranked and cuda:
        partial = []
        for variant in seeds:
            def evaluate(ngl, variant=variant):
                config = replace(variant, ngl=ngl)
                ensure_trace(config, "diverse partial fallback seed")
                result = execute_stage(config, "partial-feasibility", lambda: trial(ex, cap, model["path"], config, smoke_work, benchmark=True, device_mapping=mapping_for(config)))
                if result.get("status") == "ok": partial.append((config, score(result))); return True
                return False
            bounded_layers(maximum, None, evaluate, 4 if workload.budget == "quick" else 6)
        ranked = stratified_select(partial, plan.coarse_survivors, config_of=lambda row: row[0], score_of=lambda row: row[1])
        reasoning["partial_ngl"] = "used diverse placement/KV seeds because all full-offload anchors failed"
    elif not workload.explore_partial_offload:
        reasoning["partial_ngl"] = "skipped because full offload satisfies the requested workload"
    if not ranked:
        raise AutotuneError(f"No configuration passed the staged feasibility funnel; scope {scope}")

    target_work = replace(workload, repeats=plan.repeats)
    if workload.objective == "max-context":
        target_limit = min(finalists, plan.max_context_frontiers)
        target = max_context_select(
            ranked, target_limit, config_of=lambda row: row[0],
            performance_of=lambda row: row[1],
        )
        if workload.budget == "quick":
            reasoning["quick_frontier_plan"] = {
                "validation_mode": "startup_plus_smoke",
                "prompt_tokens": workload.frontier_smoke_tokens,
                "note": "candidate context is fully allocated at startup; only the request depth is bounded",
            }
        reasoning["stages"].append({
            "stage": "target-depth", "attempted": 0, "survived": len(target),
            "skipped": "max-context objective: frontier performs the configured validation mode",
            "placement_families": sorted({placement_family(row[0]) for row in target}),
        })
    else:
        target = []
        for index, (config, _) in enumerate(ranked, 1):
            progress(index, len(ranked), config, "target-depth benchmark")
            result = execute_stage(config, "target-depth", lambda c=config: coarse(ex, bench, model["path"], c, target_work) if bench else trial(ex, cap, model["path"], c, target_work, benchmark=True, device_mapping=mapping_for(c)))
            if result.get("status") not in ("ok", "request_timeout", "trial_timeout") and bench:
                result = execute_stage(config, "target-depth-server", lambda c=config: trial(ex, cap, model["path"], c, target_work, benchmark=True, device_mapping=mapping_for(c)))
                if result.get("status") == "ok":
                    ensure_trace(config, "target-depth server fallback")["reason_rejected_or_pruned"] = None
            if result.get("status") == "ok": target.append((config, score(result), result))
        target = stratified_select(target, min(finalists, plan.finalists), config_of=lambda row: row[0], score_of=lambda row: row[1])
        reasoning["stages"].append({"stage": "target-depth", "attempted": accounting["target_depth_attempts"], "survived": len(target), "placement_families": sorted({placement_family(row[0]) for row in target})})

    if workload.objective == "max-context":
        winners = []
        for config, measured in target:
            startup_result = store.previous(scope, config.json(), "startup") or {"status": "ok", "initialized": True, "score": measured}
            winners.append((config, measured, startup_result))
        reasoning["stages"].append({
            "stage": "validate", "attempted": 0, "survived": len(winners),
            "skipped": "max-context objective: frontier probe is the real startup/request validation",
        })
    else:
        winners = []
        for index, (config, _, _) in enumerate(target, 1):
            progress(index, len(target), config, "server validation")
            result = execute_stage(config, "validate", lambda c=config: trial(ex, cap, model["path"], c, target_work, benchmark=True, device_mapping=mapping_for(c)))
            if result.get("status") == "ok": winners.append((config, score(result), result))
    if not winners:
        raise AutotuneError(f"No finalist passed real server validation; scope {scope}")
    winners.sort(key=lambda row: (-row[1], placement_family(row[0]), kv_tier(row[0])))

    mode = workload.context_frontier if cuda else "off"
    if workload.objective == "performance":
        desired = plan.performance_frontiers
    elif workload.objective == "max-context":
        desired = plan.max_context_frontiers
    else:
        desired = plan.balanced_frontiers
    if mode == "off" or (mode == "auto" and workload.budget == "quick" and workload.objective == "performance"):
        desired = 0
    elif mode == "full":
        if workload.objective == "max-context":
            desired = min(len(winners), plan.max_context_frontiers)
        else:
            desired = len(winners)

    if workload.objective == "performance":
        frontier_candidates = winners[:desired]
    else:
        frontier_candidates = max_context_select(
            winners, min(desired, len(winners)),
            config_of=lambda row: row[0], performance_of=lambda row: row[1],
        )

    metadata, architecture = model.get("metadata", {}), model.get("architecture")
    declared = metadata.get(f"{architecture}.context_length")
    upper = workload.max_context or (declared if isinstance(declared, int) and declared > 0 else base.ctx * 4)
    frontier_work = workload
    frontiers, frontier_seen = {}, set()
    current_frontier = {"row": None, "parent": None}
    probe_keys = set()

    def frontier_trial(candidate, fw):
        if time.monotonic() >= tune_deadline:
            return {"status": "tune_timeout", "initialized": False, "error": f"Autotune wall-clock budget exceeded ({workload.tune_timeout:.0f}s)"}
        getattr(ex, "progress", lambda message: None)(
            f"frontier probe ctx={candidate.ctx} timeout={fw.trial_timeout:.0f}s elapsed={time.monotonic()-started:.0f}s"
        )
        parent = current_frontier["parent"] or candidate
        parent_row = current_frontier["row"] or ensure_trace(parent, "objective-aware frontier finalist")
        key = digest(candidate.json())
        force_repeat = key in frontier_seen
        frontier_seen.add(key)
        parent_row.setdefault("frontier_probes", [])
        probe_key = (parent_row["candidate_id"], candidate.ctx)
        if probe_key not in probe_keys:
            probe_keys.add(probe_key)
            accounting["frontier_probe_points"] += 1
            parent_row["frontier_probes"].append({"ctx": candidate.ctx})
        accounting["frontier_attempts"] += 1
        result = cached_trial(
            store, scope, candidate, "context-frontier",
            lambda: trial(ex, cap, model["path"], candidate, fw, benchmark=True, device_mapping=mapping_for(candidate)),
            retry_failed or force_repeat,
        )
        if not result.get("cache_reused"):
            accounting["model_load_attempts"] += 1
            accounting["frontier_model_load_attempts"] += 1
        return result

    for index, (config, _, _) in enumerate(frontier_candidates, 1):
        if time.monotonic() >= tune_deadline:
            reasoning["frontier_budget_exhausted"] = True
            break
        progress(index, len(frontier_candidates), config, "frontier search")
        row = ensure_trace(config, "objective-aware frontier finalist")
        if "context-frontier" not in row["stage_entered"]:
            row["stage_entered"].append("context-frontier")
        current_frontier["row"] = row
        current_frontier["parent"] = config
        frontiers[digest(config.json())] = search_context_frontier(
            config, frontier_work, max(base.ctx, upper), frontier_trial
        ).json()
    current_frontier["row"] = current_frontier["parent"] = None
    reasoning["frontier"] = "off" if not frontier_candidates else f"performed on {len(frontier_candidates)} diverse validated finalist(s) for objective={workload.objective}"

    if workload.objective == "max-context":
        measured_frontiers = [
            row for row in frontier_candidates
            if digest(row[0].json()) in frontiers
        ]
        if measured_frontiers:
            winner = max(
                measured_frontiers,
                key=lambda row: ((frontiers[digest(row[0].json())].get("recommended_safe_ctx") or 0), row[1]),
            )
        else:
            winner = winners[0]
    elif workload.objective == "balanced":
        def balanced_score(row):
            safe = (frontiers.get(digest(row[0].json())) or {}).get("recommended_safe_ctx") or base.ctx
            return (row[1] * (safe / base.ctx) ** 0.25, safe, row[1], placement_family(row[0]))
        winner = max(frontier_candidates or winners, key=balanced_score)
    else:
        winner = winners[0]
    config, selected_score, result = winner
    frontier = frontiers.get(digest(config.json()))
    for row in trace:
        if row["reason_rejected_or_pruned"] is None and row["candidate_id"] != ensure_trace(config, "winner")["candidate_id"]:
            row["reason_rejected_or_pruned"] = "not selected after objective ranking"
    selected_trace = ensure_trace(config, "winner"); selected_trace["reason_rejected_or_pruned"] = None
    reasoning["stages"].append({"stage": "frontier", "attempted": accounting["frontier_attempts"], "survived": len(frontier_candidates)})
    qualified_families = sorted({placement_family(row[0]) for row in winners})
    measured_frontier_rows = [row for row in frontier_candidates if digest(row[0].json()) in frontiers]
    explored_families = sorted({placement_family(row[0]) for row in measured_frontier_rows})
    qualified_kv = sorted({kv_tier(row[0]) for row in winners})
    explored_kv = sorted({kv_tier(row[0]) for row in measured_frontier_rows})
    search_coverage = {
        "max_context_search_scope": mode,
        "frontier_candidate_count": len(measured_frontier_rows),
        "frontier_candidate_families": explored_families,
        "frontier_candidate_kv_tiers": explored_kv,
        "unexplored_candidate_families": sorted(set(qualified_families) - set(explored_families)),
        "unexplored_kv_tiers": sorted(set(qualified_kv) - set(explored_kv)),
        "optimality": (
            "incomplete" if not measured_frontier_rows and workload.objective == "max-context"
            else "budget_bounded" if workload.objective == "max-context" and workload.budget == "quick"
            else "full_shortlist" if mode == "full" and len(measured_frontier_rows) == len(winners)
            else "budget_bounded"
        ),
    }
    reasoning["max_context_search_coverage"] = search_coverage
    manifest["cost_accounting"] = accounting; manifest["candidate_trace"] = trace; manifest["search_reasoning"] = reasoning; manifest["max_context_search_coverage"] = search_coverage
    store.write("profiles", scope + "-manifest", manifest)
    profile = {"scope": scope, "identity": identity, "model_path": model["path"], "model_fingerprint": model["fingerprint"], "hardware_fingerprint": env["fingerprint"], "server_fingerprint": cap.fingerprint, "server_path": cap.path, "config": config.json(), "workload": asdict(workload), "score": selected_score, "validation": result, "depth": depth.json(), "device_mapping": result.get("device_mapping"), "placement_evidence": result.get("placement_evidence"), "context_frontier": frontier, "max_allocatable_ctx": frontier.get("max_allocatable_ctx") if frontier else config.ctx, "max_allocatable_ctx_is_lower_bound": frontier.get("max_allocatable_ctx_is_lower_bound") if frontier else True, "max_benchmarkable_ctx": frontier.get("max_benchmarkable_ctx") if frontier else config.ctx, "recommended_safe_ctx": frontier.get("recommended_safe_ctx") if frontier else None, "decision": f"Selected by objective={workload.objective} after adaptive diversified full-offload search.", "search_reasoning": reasoning, "max_context_search_coverage": search_coverage, "cost_accounting": accounting, "candidate_trace": trace, "created": time.time(), "manifest": str(store.root / "profiles" / (scope + "-manifest.json"))}
    store.write("profiles", scope, profile); store.write("profiles", digest(model["path"]) + "-latest", profile)
    return profile
