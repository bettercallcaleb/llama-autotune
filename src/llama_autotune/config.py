from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import os

SCHEMA = 1
POLICY = "v0.1.8-smoke-frontier-1"


class AutotuneError(RuntimeError):
    pass


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def cache_root() -> Path:
    return (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "llama-autotune"
    )


@dataclass(frozen=True)
class RuntimeConfig:
    ngl: int
    ctk: str = "f16"
    ctv: str = "f16"
    flash: str = "off"
    ctx: int = 4096
    parallel: int = 1
    batch: int = 512
    ubatch: int = 128
    split_mode: str | None = None
    tensor_split: str | None = None
    devices: str | None = None

    def json(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Workload:
    prompt: str = (
        "Explain how a computer processes instructions, with a concrete example. " * 32
    )
    tokens: int = 64
    repeats: int = 2
    startup_timeout: float = 180
    request_timeout: float = 300
    trial_timeout: float = 60
    tune_timeout: float = 600
    headroom_mb: int = 512
    headroom_percent: float = 5
    depth: str = "auto"
    depth_ratio: float = 0.75
    context_frontier: str = "auto"
    max_context: int | None = None
    context_resolution: int = 4096
    context_headroom_tokens: int = 8192
    context_headroom_percent: float = 5
    frontier_reserve_tokens: int = 256
    frontier_smoke_tokens: int = 2048
    budget: str = "normal"
    objective: str = "performance"
    explore_partial_offload: bool = False
