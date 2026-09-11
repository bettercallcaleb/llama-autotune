from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import math
import re
import shutil
from ..config import AutotuneError, RuntimeConfig, digest
from ..executor import Executor


@dataclass
class Capabilities:
    path: str
    version: str
    help: str
    flags: dict[str, str]
    devices: str
    fingerprint: str

    def has(self, *names: str) -> bool:
        return any(n in self.flags for n in names)

    def flag(self, *names: str) -> str:
        for n in names:
            if n in self.flags:
                return n
        raise AutotuneError(f"{self.path} does not advertise any of {names}")

    def json(self) -> dict:
        return asdict(self)


def parse_help(text: str) -> dict[str, str]:
    flags = {}
    current = []
    for line in text.splitlines():
        declaration = re.match(
            r"^\s*(--?[A-Za-z][A-Za-z0-9-]*(?:,\s*--?[A-Za-z][A-Za-z0-9-]*)*)", line
        )
        if declaration:
            current = re.findall(r"--?[A-Za-z][A-Za-z0-9-]*", declaration.group(1))
            for flag in current:
                flags[flag] = line.strip()
        else:
            for flag in current:
                flags[flag] += " " + line.strip()
    return flags


def discover(path: Path, ex: Executor) -> Capabilities:
    result = ex.run([path, "--help"], timeout=30, check=True)
    text = result.stdout + result.stderr
    flags = parse_help(text)
    if not flags:
        raise AutotuneError(f"Unrecognized help output from {path}")
    version_result = ex.run([path, "--version"], timeout=20)
    version = version_result.stdout + version_result.stderr
    devices = ""
    if "--list-devices" in flags:
        result = ex.run([path, "--list-devices"], timeout=30)
        devices = result.stdout + result.stderr
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    libraries = []
    for directory in sorted({path.parent, path.parent.parent / "lib"}):
        for lib in sorted(directory.glob("*.so*")):
            s = lib.stat()
            libraries.append(
                (str(lib.resolve()), s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            )
    linked = ex.run(["ldd", path], timeout=15)
    if linked.ok:
        for name in re.findall(r"=>\s+(/\S+)", linked.stdout):
            lib = Path(name)
            try:
                s = lib.stat()
                libraries.append(
                    (str(lib.resolve()), s.st_size, s.st_mtime_ns, s.st_ctime_ns)
                )
            except OSError:
                pass
    fp = digest(
        {
            "binary": h.hexdigest(),
            "version": version,
            "help": text,
            "libraries": libraries,
        }
    )
    return Capabilities(str(path), version, text, flags, devices, fp)


def find_binary(name: str, explicit: str | None, build_dir: str | None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.is_file():
            raise AutotuneError(f"Binary does not exist: {p}")
        return p
    if build_dir:
        root = Path(build_dir).expanduser().resolve()
        for parent in (root / "bin", root / "bin" / "Release", root):
            for suffix in ("", ".exe"):
                p = parent / (name + suffix)
                if p.is_file():
                    return p
        return None
    path = shutil.which(name)
    return Path(path).resolve() if path else None


def flash_args(cap: Capabilities, value: str) -> list[str]:
    if not cap.has("--flash-attn", "-fa"):
        if value == "off":
            return []
        raise AutotuneError("Flash attention not advertised")
    flag = cap.flag("--flash-attn", "-fa")
    line = cap.flags[flag]
    if re.search(r"on\s*\|\s*off", line):
        return [flag, value]
    if re.search(r"0\s*\|\s*1", line):
        return [flag, "1" if value == "on" else "0"]
    if re.search(r"[<\[].*[>\]]", line):
        raise AutotuneError("Unrecognized flash-attention value syntax")
    return [flag] if value == "on" else []


def runtime_args(
    cap: Capabilities,
    model: str,
    c: RuntimeConfig,
    server: bool = True,
    fitting: bool = False,
    margin: int | str = 512,
) -> list[str]:
    args = [cap.path, cap.flag("--model", "-m"), model]
    if not fitting:
        args += [cap.flag("--gpu-layers", "--n-gpu-layers", "-ngl"), str(c.ngl)]
    for names, value in [
        (("--cache-type-k", "-ctk"), c.ctk),
        (("--cache-type-v", "-ctv"), c.ctv),
    ]:
        if cap.has(*names):
            args += [cap.flag(*names), value]
        elif value != "f16":
            raise AutotuneError(f"Unsupported KV type option {names}")
    args += flash_args(cap, c.flash)
    for names, value in [
        (("--batch-size", "-b"), c.batch),
        (("--ubatch-size", "-ub"), c.ubatch),
    ]:
        args += [cap.flag(*names), str(value)]
    for names, value in [
        (("--split-mode", "-sm"), c.split_mode),
        (("--tensor-split", "-ts"), c.tensor_split),
        (("--device", "-dev"), c.devices),
    ]:
        if value is not None:
            args += [cap.flag(*names), value]
    if server:
        args += [
            cap.flag("--ctx-size", "-c"),
            str(c.ctx),
            cap.flag("--parallel", "-np"),
            str(c.parallel),
        ]
        if cap.has("--fit", "-fit"):
            args += [cap.flag("--fit", "-fit"), "on" if fitting else "off"]
        if fitting:
            args += [cap.flag("--fit-target", "-fitt"), str(margin)]
        if cap.has("--no-warmup"):
            args += ["--no-warmup"]
    return args


def parse_offload(text: str) -> int | None:
    values = re.findall(r"offloaded\s+(\d+)\s*/\s*\d+\s+layers", text)
    return int(values[-1]) if values else None


def parse_bench(text: str) -> dict:
    try:
        rows = json.loads(text)
        if not isinstance(rows, list):
            raise ValueError("Expected JSON array")
        pp = [
            float(r["avg_ts"])
            for r in rows
            if int(r.get("n_prompt", 0)) > 0 and int(r.get("n_gen", 0)) == 0
        ]
        tg = [
            float(r["avg_ts"])
            for r in rows
            if int(r.get("n_gen", 0)) > 0 and int(r.get("n_prompt", 0)) == 0
        ]
        if not pp or not tg or not all(math.isfinite(x) and x > 0 for x in pp + tg):
            raise ValueError("Missing valid prompt/decode measurements")
        return {"pp_ts": sum(pp) / len(pp), "tg_ts": sum(tg) / len(tg), "rows": rows}
    except (ValueError, TypeError, KeyError) as exc:
        raise AutotuneError(f"Malformed llama-bench JSON: {exc}") from exc
