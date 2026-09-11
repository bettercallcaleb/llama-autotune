from dataclasses import dataclass, replace
from typing import Callable, Iterable
from ..config import RuntimeConfig
from ..llama.capabilities import Capabilities
from ..llama.devices import map_devices, parse_devices


@dataclass(frozen=True)
class BudgetPolicy:
    placements: int
    max_initial_model_loads: int
    startup_survivors: int
    smoke_survivors: int
    coarse_survivors: int
    finalists: int
    repeats: int
    smoke_depth: int
    performance_frontiers: int
    max_context_frontiers: int
    balanced_frontiers: int
    partial_seeds: int


BUDGETS = {
    "quick": BudgetPolicy(2, 6, 4, 4, 2, 1, 1, 256, 0, 1, 1, 2),
    "normal": BudgetPolicy(3, 9, 7, 5, 4, 3, 2, 1024, 1, 2, 2, 3),
    "thorough": BudgetPolicy(5, 18, 14, 12, 8, 4, 4, 2048, 3, 3, 4, 5),
}


def budget_policy(name: str) -> BudgetPolicy:
    return BUDGETS[name]


def placement_family(config: RuntimeConfig) -> str:
    if config.devices == "none": return "cpu"
    if config.tensor_split: return f"tensor-split:{config.devices}:{config.tensor_split}"
    if config.split_mode: return f"{config.split_mode}-auto:{config.devices}"
    return f"single:{config.devices or 'default'}"


def kv_tier(config: RuntimeConfig) -> str:
    return f"{config.ctk}/{config.ctv}"


def kv_efficiency(config: RuntimeConfig) -> int:
    rank = {"q4_0": 3, "q8_0": 2, "f16": 1}
    return rank.get(config.ctk, 0) + rank.get(config.ctv, 0)


def placement_capacity_class(config: RuntimeConfig) -> int:
    if config.tensor_split: return 3
    if config.devices and "," in config.devices: return 2
    return 1


def _mapped_free_weights(ids, cap, env, workload):
    selected = ",".join(ids)
    mapping = map_devices(cap.devices, env.get("gpus", []), selected, workload, env.get("visibility", {}))
    physical = {gpu.get("uuid"): gpu for gpu in env.get("gpus", [])}
    weights = []
    limitations = list(mapping.diagnostics)
    for identifier in ids:
        entry = next((row for row in mapping.entries if row["llama_device"] == identifier), None)
        gpu = physical.get(entry.get("uuid")) if entry else None
        if gpu and mapping.reliable:
            weights.append(max(1.0, float(gpu.get("free_mb", 0))))
        else:
            advertised = next(device for device in parse_devices(cap.devices) if device.id == identifier)
            weights.append(max(1.0, advertised.total_mb))
            limitations.append(f"{identifier}: reliable free-memory mapping unavailable; advertised capacity used")
    return weights, sorted(set(limitations))


def placement_candidates(base: RuntimeConfig, cap: Capabilities, env: dict, workload, limit: int) -> tuple[list[RuntimeConfig], list[str]]:
    if base.devices: return [base], []
    devices = parse_devices(cap.devices)
    if not devices or not cap.has("--device", "-dev"): return [base], ["No parsed device inventory; placement expansion unavailable"]
    ordered = sorted(devices, key=lambda device: device.total_mb, reverse=True)
    primary = ordered[0]
    rows = [replace(base, devices=primary.id, split_mode=None, tensor_split=None)]
    limitations = []
    if len(ordered) > 1:
        ids_list = [device.id for device in ordered]
        selected = ",".join(ids_list)
        rows.append(replace(base, devices=selected, split_mode="layer", tensor_split=None))
        if cap.has("--tensor-split", "-ts") and cap.has("--split-mode", "-sm"):
            weights, notes = _mapped_free_weights(ids_list, cap, env, workload)
            limitations.extend(notes)
            minimum = min(weights)
            free_ratio = [max(1, round(weight / minimum)) for weight in weights]
            capacity_minimum = min(device.total_mb for device in ordered)
            capacity_ratio = [max(1, round(device.total_mb / capacity_minimum)) for device in ordered]
            ratios = [free_ratio, capacity_ratio]
            if len(ordered) == 2:
                base_ratio = capacity_ratio[0]
                ratios.extend(([max(1, base_ratio - 1), 1], [base_ratio + 1, 1]))
            seen = set()
            for ratio in ratios:
                value = ",".join(map(str, ratio))
                if value not in seen:
                    seen.add(value); rows.append(replace(base, devices=selected, split_mode="layer", tensor_split=value))
    return rows[:limit], limitations


def _max_context_anchor_rows(base: RuntimeConfig, placements: list[RuntimeConfig], tiers: list[tuple[str, str]], limit: int, preferred_fa: str) -> list[RuntimeConfig]:
    primary = next((p for p in placements if not p.tensor_split and not (p.devices and "," in p.devices)), placements[0])
    explicit = next((p for p in placements if p.tensor_split), None)
    auto_multi = next((p for p in placements if not p.tensor_split and p.devices and "," in p.devices), None)
    tier_map = {k: (k, v) for k, v in tiers}
    rows: list[RuntimeConfig] = []
    def add(place, kind):
        if place is None or kind not in tier_map or len(rows) >= limit: return
        k, v = tier_map[kind]
        candidate = replace(place, ctk=k, ctv=v, flash=preferred_fa)
        if candidate not in rows: rows.append(candidate)
    for kind in ("f16", "q8_0", "q4_0"): add(primary, kind)
    for kind in ("q4_0", "q8_0"): add(explicit, kind)
    add(auto_multi, "q4_0")
    for kind in ("q4_0", "q8_0", "f16"):
        for place in placements: add(place, kind)
    return rows[:limit]


def variants(base: RuntimeConfig, cap: Capabilities, cuda: bool, budget: str = "thorough", env: dict | None = None, workload=None) -> list[RuntimeConfig]:
    if not cuda: return [replace(base, ngl=0, devices="none" if cap.has("--device", "-dev") else None)]
    from ..config import Workload
    plan = budget_policy(budget)
    workload = workload or Workload()
    placement_limit = max(plan.placements, 3) if workload.objective == "max-context" else plan.placements
    placements, _ = placement_candidates(base, cap, env or {}, workload, placement_limit)
    ks = ["f16", "q8_0", "q4_0"] if cap.has("--cache-type-k", "-ctk") else ["f16"]
    vs = ["f16", "q8_0", "q4_0"] if cap.has("--cache-type-v", "-ctv") else ["f16"]
    tiers = [(kind, kind) for kind in ("f16", "q8_0", "q4_0") if kind in ks and kind in vs]
    preferred_fa = "on" if cap.has("--flash-attn", "-fa") else "off"
    if workload.objective == "max-context": return _max_context_anchor_rows(base, placements, tiers, plan.max_initial_model_loads, preferred_fa)
    rows = [replace(place, ctk=k, ctv=v, flash=preferred_fa) for k, v in tiers for place in placements]
    return rows[:plan.max_initial_model_loads]


def alternate_fa(config: RuntimeConfig, cap: Capabilities) -> RuntimeConfig | None:
    if not cap.has("--flash-attn", "-fa"): return None
    return replace(config, flash="off" if config.flash == "on" else "on")


def stratified_select(rows: Iterable, limit: int, config_of=lambda row: row, score_of=lambda row: 0.0) -> list:
    rows = list(rows)
    if len(rows) <= limit: return rows
    selected, used = [], set()
    def add_best(key):
        groups = {}
        for index, row in enumerate(rows): groups.setdefault(key(config_of(row)), []).append((index, row))
        for _, choices in groups.items():
            choices.sort(key=lambda item: (-score_of(item[1]), item[0]))
            index, row = choices[0]
            if index not in used and len(selected) < limit: used.add(index); selected.append(row)
    add_best(placement_family); add_best(kv_tier); add_best(lambda config: config.flash)
    remainder = sorted(enumerate(rows), key=lambda item: (-score_of(item[1]), item[0]))
    for index, row in remainder:
        if index not in used and len(selected) < limit: used.add(index); selected.append(row)
    return selected


def max_context_select(rows: Iterable, limit: int, config_of=lambda row: row, performance_of=lambda row: 0.0) -> list:
    rows = list(rows)
    if len(rows) <= limit: return rows
    selected: list = []; used: set[int] = set()
    def rank(index_row):
        index, row = index_row; cfg = config_of(row)
        return (-kv_efficiency(cfg), -placement_capacity_class(cfg), -float(performance_of(row) or 0), index)
    def add_best(predicate, prefer_different_placement=False):
        if len(selected) >= limit: return
        current_families = {placement_family(config_of(row)) for row in selected}
        choices = [(i, row) for i, row in enumerate(rows) if i not in used and predicate(config_of(row))]
        if prefer_different_placement:
            different = [(i, row) for i, row in choices if placement_family(config_of(row)) not in current_families]
            if different: choices = different
        if not choices: return
        i, row = sorted(choices, key=rank)[0]; used.add(i); selected.append(row)
    add_best(lambda c: c.ctk == "q4_0" and c.ctv == "q4_0")
    add_best(lambda c: c.ctk == "q8_0" and c.ctv == "q8_0", prefer_different_placement=True)
    if not any(config_of(row).tensor_split for row in selected): add_best(lambda c: bool(c.tensor_split))
    if not any(not (config_of(row).devices and "," in config_of(row).devices) for row in selected): add_best(lambda c: not (c.devices and "," in c.devices))
    add_best(lambda c: c.ctk == "f16" and c.ctv == "f16", prefer_different_placement=True)
    for i, row in sorted(enumerate(rows), key=rank):
        if i not in used and len(selected) < limit: used.add(i); selected.append(row)
    return selected


def time_bounded_capacity_select(rows: Iterable, limit: int, next_ctx: int, trial_timeout: float, metrics_of, config_of=lambda row: row, performance_of=lambda row: 0.0, reserve_tokens: int = 320):
    rows = list(rows); diagnostics = []; viable = []
    for row in rows:
        pp_ts, tg_ts = metrics_of(row); predicted = None
        if pp_ts and pp_ts > 0:
            prompt_tokens = max(1, next_ctx - reserve_tokens); predicted = prompt_tokens / float(pp_ts)
            if tg_ts and tg_ts > 0: predicted += 64.0 / float(tg_ts)
        diagnostics.append({"placement": placement_family(config_of(row)), "kv": kv_tier(config_of(row)), "pp_ts": pp_ts, "predicted_next_probe_seconds": predicted, "within_trial_budget": predicted is not None and predicted <= trial_timeout})
        if predicted is not None and predicted <= trial_timeout: viable.append(row)
    pool = viable if viable else rows
    return max_context_select(pool, min(limit, len(pool)), config_of=config_of, performance_of=performance_of), diagnostics


def diverse_partial_seeds(rows: Iterable[RuntimeConfig], limit: int) -> list[RuntimeConfig]:
    efficient = sorted(rows, key=lambda config: ({"q4_0": 0, "q8_0": 1, "f16": 2}.get(config.ctk, 3), placement_family(config)))
    return stratified_select(efficient, limit, score_of=lambda config: -({"q4_0": 0, "q8_0": 1, "f16": 2}.get(config.ctk, 3)))


def bounded_layers(maximum: int, seed: int | None, evaluate: Callable[[int], bool], budget: int = 6) -> list[int]:
    seen: dict[int, bool] = {}
    def test(n: int) -> bool:
        n = min(maximum, max(0, n))
        if n not in seen: seen[n] = evaluate(n)
        return seen[n]
    low, high = 0, maximum
    if test(maximum): return [maximum]
    high = maximum - 1
    if seed is not None and len(seen) < budget:
        if test(seed): low = seed
        else: high = min(high, seed - 1)
    while low < high and len(seen) < max(1, budget - 1):
        mid = (low + high + 1) // 2
        if test(mid): low = mid
        else: high = mid - 1
    if len(seen) < budget: test(low)
    return [n for n, ok in seen.items() if ok]
