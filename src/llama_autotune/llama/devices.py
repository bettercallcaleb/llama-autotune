from dataclasses import asdict, dataclass
import math
import re
from ..config import AutotuneError, Workload


@dataclass(frozen=True)
class LlamaDevice:
    id: str
    name: str
    total_mb: float
    uuid: str | None
    pci_bus: str | None
    evidence: str


@dataclass(frozen=True)
class DeviceMapping:
    selected_ids: list[str]
    entries: list[dict]
    monitor_uuids: list[str] | None
    fit_target: str | None
    reliable: bool
    diagnostics: list[str]

    def json(self) -> dict:
        return asdict(self)


def normalize_pci(value: str) -> str:
    pieces = value.lower().split(":")
    if len(pieces) == 2:
        pieces.insert(0, "0000")
    return ":".join([pieces[0][-4:].zfill(4), *pieces[1:]])


def parse_devices(text: str) -> list[LlamaDevice]:
    devices = []
    seen = set()
    for line in text.splitlines():
        match = re.match(
            r"^\s*(CUDA\d+):\s*(.*?)\s*\((\d+(?:\.\d+)?) MiB,\s*\d+(?:\.\d+)? MiB free\)",
            line,
        )
        if not match:
            continue
        identifier, name, total = match.groups()
        if identifier in seen:
            raise AutotuneError("Duplicate llama.cpp device identifier")
        seen.add(identifier)
        uuid = re.search(r"GPU-[A-Za-z0-9-]+", line)
        pci = re.search(
            r"(?:[0-9a-fA-F]{4,8}:)?[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", line
        )
        name = re.sub(r"\s*\[.*?\]", "", name).strip()
        devices.append(
            LlamaDevice(
                identifier,
                name,
                float(total),
                uuid[0] if uuid else None,
                normalize_pci(pci[0]) if pci else None,
                line.strip(),
            )
        )
    return devices


def headroom_mb(gpu: dict, workload: Workload) -> int:
    total = float(gpu["total_mb"])
    if not math.isfinite(total) or total <= 0:
        raise AutotuneError("Cannot calculate margin without valid total VRAM")
    return math.ceil(max(workload.headroom_mb, total * workload.headroom_percent / 100))


def _name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def map_devices(
    text: str,
    gpus: list[dict],
    selected: str | None,
    workload: Workload,
    visibility: dict | None = None,
) -> DeviceMapping:
    if selected == "none":
        return DeviceMapping([], [], [], None, True, [])
    advertised = parse_devices(text)
    by_id = {d.id: d for d in advertised}
    ids = (
        [x.strip() for x in selected.split(",")]
        if selected
        else [d.id for d in advertised]
    )
    if len(ids) != len(set(ids)):
        raise AutotuneError("Repeated --device identifier")
    if selected and advertised and any(i not in by_id for i in ids):
        raise AutotuneError(
            "Selected device is not an advertised CUDA device; v0.1.5 supports CUDA/CPU only"
        )
    if not advertised:
        return DeviceMapping(
            ids,
            [],
            None,
            None,
            False,
            [
                "Cannot parse llama.cpp CUDA device inventory; fitting disabled and all NVIDIA devices monitored"
            ],
        )
    physical = {g["uuid"]: g for g in gpus}
    visible = (visibility or {}).get("CUDA_VISIBLE_DEVICES") or ""
    visible_ids = [v.strip() for v in visible.split(",")]
    entries = []
    used = set()
    for identifier in ids:
        dev = by_id[identifier]
        candidates = []
        method = "unresolved"
        if dev.uuid:
            candidates = [u for u in physical if u == dev.uuid]
            method = "advertised_uuid"
        elif dev.pci_bus:
            candidates = [
                u
                for u, g in physical.items()
                if normalize_pci(g.get("pci_bus", "")) == dev.pci_bus
            ]
            method = "advertised_pci"
        else:
            ordinal = int(identifier[4:])
            if ordinal < len(visible_ids) and visible_ids[ordinal].startswith("GPU-"):
                candidates = [u for u in physical if u.startswith(visible_ids[ordinal])]
                method = "CUDA_VISIBLE_DEVICES_UUID_order"
            else:
                candidates = [
                    u
                    for u, g in physical.items()
                    if _name(g["name"]) == _name(dev.name)
                    and abs(g["total_mb"] - dev.total_mb)
                    <= max(64, g["total_mb"] * 0.05)
                ]
                method = (
                    "unique_name_and_capacity"
                    if len(candidates) == 1
                    else "equivalent_name_and_capacity"
                )
        mapped = (
            candidates[0]
            if len(candidates) == 1 and candidates[0] not in used
            else None
        )
        if mapped:
            used.add(mapped)
        entries.append(
            {
                "llama_device": identifier,
                "name": dev.name,
                "llama_total_mb": dev.total_mb,
                "uuid": mapped,
                "candidate_uuids": sorted(candidates),
                "method": method,
                "margin_mb": (
                    headroom_mb(physical[mapped], workload) if mapped else None
                ),
                "evidence": dev.evidence,
            }
        )
    unresolved = [e for e in entries if e["uuid"] is None]
    diagnostics = []
    if unresolved:
        union = set().union(*(set(e["candidate_uuids"]) for e in unresolved))
        uniform = (
            set(ids) == set(by_id)
            and len(union) == len(unresolved)
            and not union.intersection(used)
            and all(set(e["candidate_uuids"]) == union for e in unresolved)
            and len({physical[u]["total_mb"] for u in union}) == 1
        )
        if uniform:
            margin = headroom_mb(physical[next(iter(union))], workload)
            for entry in unresolved:
                entry["margin_mb"] = margin
                entry["method"] = "all_identical_devices_order_independent"
            diagnostics.append(
                "Identical devices cannot be distinguished physically; all are selected, margins are identical, and the whole equivalence group is monitored"
            )
        else:
            diagnostics.append(
                "Ambiguous CUDA-to-NVIDIA mapping; per-device fitting disabled and all NVIDIA devices conservatively monitored"
            )
            return DeviceMapping(ids, entries, None, None, False, diagnostics)
    monitored = sorted(set().union(*(set(e["candidate_uuids"]) for e in entries)))
    target = ",".join(str(e["margin_mb"]) for e in entries)
    return DeviceMapping(ids, entries, monitored, target, not unresolved, diagnostics)
