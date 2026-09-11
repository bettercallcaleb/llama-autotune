from pathlib import Path
from typing import Callable
import csv
import io
import json
import math
import os
import platform
import re
import shutil
import time
from ..config import digest
from ..executor import Executor


def read_text(path: str) -> str:
    try:
        return Path(path).read_text(errors="replace")
    except OSError:
        return ""


def parse_nvidia(text: str) -> list[dict]:
    result = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 6:
            continue
        try:
            total, free = float(row[2]), float(row[3])
            if (
                not math.isfinite(total)
                or not math.isfinite(free)
                or total <= 0
                or not 0 <= free <= total
            ):
                continue
            result.append(
                {
                    "vendor": "NVIDIA",
                    "uuid": row[0].strip(),
                    "name": row[1].strip(),
                    "total_mb": total,
                    "free_mb": free,
                    "driver": row[4].strip(),
                    "pci_bus": row[5].strip(),
                    "compute_capability": row[6].strip() if len(row) > 6 else None,
                }
            )
        except ValueError:
            continue
    return result


def nvidia_memory(executor: Executor) -> list[dict]:
    args = [
        "nvidia-smi",
        "--query-gpu=uuid,name,memory.total,memory.free,driver_version,pci.bus_id,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    r = executor.run(args, timeout=10, quiet=True)
    if not r.ok:
        args[1] = args[1].replace(",compute_cap", "")
        r = executor.run(args, timeout=10, quiet=True)
    if not r.ok:
        return []
    rows = parse_nvidia(r.stdout)
    raw_rows = [row for row in csv.reader(io.StringIO(r.stdout)) if row]
    return rows if len(rows) == len(raw_rows) else []


def parse_lscpu(text: str) -> dict:
    try:
        return {r["field"].rstrip(":"): r["data"] for r in json.loads(text)["lscpu"]}
    except (ValueError, KeyError, TypeError):
        return {}


VERSION_TOOLS = {
    "nvcc",
    "hipcc",
    "icpx",
    "gcc",
    "clang",
    "cmake",
    "ninja",
    "ccache",
    "git",
    "nvidia-smi",
}


def executable_identity(location: str) -> dict:
    path = Path(location).resolve()
    try:
        st = path.stat()
        return {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    except OSError:
        return {"path": str(path), "size": None, "mtime_ns": None}


def parse_numa(text: str) -> dict:
    nodes = {}
    distances = []
    for line in text.splitlines():
        match = re.fullmatch(r"node (\d+) (cpus|size):\s*(.*)", line.strip())
        if match:
            node, field, value = match.groups()
            nodes.setdefault(node, {})[field] = value.split()
        elif re.fullmatch(r"\s*\d+:\s*(?:\d+\s*)+", line):
            distances.append(line.split())
    return {"nodes": nodes, "distances": distances}


def parse_pci(text: str) -> list[dict]:
    devices = []
    for line in text.splitlines():
        match = re.match(r"^([0-9a-fA-F:.]+)\s+(.+)$", line)
        if match and ":" in match[1]:
            devices.append({"bus": match[1].lower(), "description": match[2].strip()})
    return sorted(devices, key=lambda d: d["bus"])


def stable_environment(env: dict) -> dict:
    return {
        "os": env["os"],
        "cpu": {
            k: env["cpu"].get(k)
            for k in ("model", "logical", "physical", "instructions", "affinity")
        },
        "memory_total_kb": env["memory"].get("MemTotal"),
        "gpus": sorted(
            [
                {
                    k: g.get(k)
                    for k in (
                        "vendor",
                        "uuid",
                        "name",
                        "total_mb",
                        "driver",
                        "pci_bus",
                        "compute_capability",
                    )
                }
                for g in env["gpus"]
            ],
            key=lambda g: g["uuid"] or "",
        ),
        "inventory": env.get("inventory", {}),
        "tool_identities": {
            k: {"executable": v.get("identity"), "version": v.get("version_id")}
            for k, v in env["tools"].items()
        },
        "visibility": env["visibility"],
        "container_limits": env["container_limits"],
    }


def probe(
    executor: Executor,
    reader: Callable[[str], str] = read_text,
    which: Callable[[str], str | None] = shutil.which,
) -> dict:
    cpu = parse_lscpu(executor.run(["lscpu", "--json"]).stdout)
    mem = {}
    for line in reader("/proc/meminfo").splitlines():
        bits = line.split()
        if len(bits) >= 2 and bits[1].isdigit():
            mem[bits[0].rstrip(":")] = int(bits[1])
    tools = {}
    evidence = {}
    commands = {
        "nvcc": ["--version"],
        "hipcc": ["--version"],
        "rocminfo": [],
        "vulkaninfo": ["--summary"],
        "sycl-ls": [],
        "icpx": ["--version"],
        "gcc": ["--version"],
        "clang": ["--version"],
        "cl": [],
        "cmake": ["--version"],
        "ninja": ["--version"],
        "ccache": ["--version"],
        "git": ["--version"],
        "nvidia-smi": ["--version"],
        "lspci": ["-nn"],
        "numactl": ["--hardware"],
    }
    for name, flags in commands.items():
        location = which(name)
        if location:
            r = executor.run([location, *flags], timeout=15)
            raw = r.stdout + r.stderr
            version_id = (
                sorted(set(re.findall(r"\b\d+(?:\.\d+){1,3}(?:[-+][\w.]+)?\b", raw)))
                if name in VERSION_TOOLS and r.ok
                else None
            )
            tools[name] = {
                "path": location,
                "identity": executable_identity(location),
                "version": version_id,
                "version_id": version_id,
                "ok": r.ok,
                "log": r.log,
            }
            evidence[name] = {"stdout": r.stdout, "stderr": r.stderr, "log": r.log}
        else:
            tools[name] = {"path": None, "version": None, "ok": False}
    gpus = nvidia_memory(executor) if which("nvidia-smi") else []
    topology = executor.run(["nvidia-smi", "topo", "-m"]).stdout if gpus else ""
    drm = []
    for path in Path("/sys/class/drm").glob("card[0-9]*/device/vendor"):
        drm.append(
            {
                "vendor_id": reader(str(path)).strip(),
                "device_id": reader(str(path.parent / "device")).strip(),
            }
        )
    env = {
        "timestamp": time.time(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
            "wsl": "microsoft" in platform.release().lower(),
            "container": Path("/.dockerenv").exists()
            or "container" in reader("/proc/1/cgroup")
            or Path("/run/.containerenv").exists(),
        },
        "cpu": {
            "model": cpu.get("Model name", platform.processor()),
            "logical": os.cpu_count(),
            "physical": None,
            "details": cpu,
            "instructions": cpu.get("Flags", cpu.get("Features", "")).split(),
            "affinity": (
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
        },
        "memory": mem,
        "gpus": gpus,
        "other_gpu_evidence": drm,
        "topology": topology,
        "tools": tools,
        "metal": platform.system() == "Darwin",
        "visibility": {
            k: os.environ.get(k)
            for k in (
                "CUDA_VISIBLE_DEVICES",
                "CUDA_DEVICE_ORDER",
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "GGML_CUDA_ENABLE_UNIFIED_MEMORY",
                "LD_LIBRARY_PATH",
                "LD_PRELOAD",
                "OMP_NUM_THREADS",
                "GGML_CUDA_FORCE_MMQ",
                "GGML_CUDA_FORCE_CUBLAS",
            )
        },
        "container_limits": {
            p: reader("/sys/fs/cgroup/" + p).strip()
            for p in (
                "memory.max",
                "memory.swap.max",
                "cpu.max",
                "cpuset.cpus.effective",
            )
        },
        "diagnostics": [],
    }
    try:
        env["cpu"]["physical"] = int(cpu["Socket(s)"]) * int(cpu["Core(s) per socket"])
    except (KeyError, ValueError, TypeError):
        pass
    if not gpus:
        env["diagnostics"].append(
            "No queryable NVIDIA GPU; CPU remains available. Other accelerators are probe-only in v0."
        )
    if platform.system() != "Linux":
        env["diagnostics"].append(
            "v0 build/tune/run is Linux-only; some probe fields are unavailable."
        )
    numa_raw = evidence.get("numactl", {}).get("stdout", "")
    env["inventory"] = {
        "numa": parse_numa(numa_raw),
        "pci": parse_pci(evidence.get("lspci", {}).get("stdout", "")),
        "drm": sorted(drm, key=lambda d: (d["vendor_id"], d["device_id"])),
        "gpu_topology": [
            line.split()
            for line in topology.splitlines()
            if re.match(r"^\s*(GPU\d+|NIC\d+)\s", line)
        ],
    }
    env["telemetry"] = {
        "memory_kb": mem,
        "gpu_free_mb": {g["uuid"]: g["free_mb"] for g in gpus},
        "numa_free_mb": {
            n: int(v) for n, v in re.findall(r"node (\d+) free:\s*(\d+) MB", numa_raw)
        },
    }
    env["evidence"] = {"tools": evidence, "lscpu": cpu, "gpu_topology": topology}
    env["fingerprint"] = digest(stable_environment(env))
    return env
