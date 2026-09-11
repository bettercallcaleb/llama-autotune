from collections import Counter
from pathlib import Path
import hashlib
import re
import struct
from ..config import AutotuneError, digest
from ..storage.store import Store

FORMATS = {
    0: "B",
    1: "b",
    2: "H",
    3: "h",
    4: "I",
    5: "i",
    6: "f",
    7: "?",
    10: "Q",
    11: "q",
    12: "d",
}


class Reader:
    def __init__(self, file):
        self.file = file
        self.size = file.seek(0, 2)
        file.seek(0)

    def take(self, n: int) -> bytes:
        if n < 0 or n > 16_000_000 or self.file.tell() + n > self.size:
            raise AutotuneError("Malformed/truncated GGUF field")
        data = self.file.read(n)
        if len(data) != n:
            raise AutotuneError("Truncated GGUF")
        return data

    def number(self, fmt: str):
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))[0]

    def string(self, keep: bool = True):
        size = self.number("Q")
        if keep:
            return self.take(size).decode("utf-8", "replace")
        self.skip(size)
        return None

    def skip(self, n: int) -> None:
        if n < 0 or self.file.tell() + n > self.size:
            raise AutotuneError("GGUF field extends beyond file")
        self.file.seek(n, 1)

    def value(self, kind: int, keep: bool = True, depth: int = 0):
        if depth > 3:
            raise AutotuneError("Excessively nested GGUF metadata")
        if kind in FORMATS:
            return self.number(FORMATS[kind])
        if kind == 8:
            return self.string(keep)
        if kind == 9:
            subtype, count = self.number("I"), self.number("Q")
            if count > 10_000_000:
                raise AutotuneError("GGUF array exceeds metadata safety limit")
            if subtype in FORMATS:
                self.skip(count * struct.calcsize("<" + FORMATS[subtype]))
            else:
                for _ in range(count):
                    self.value(subtype, False, depth + 1)
            return {"array_type": subtype, "count": count}
        raise AutotuneError(f"Unknown GGUF metadata type {kind}")


def metadata(path: Path) -> dict:
    with path.open("rb") as f:
        r = Reader(f)
        if r.take(4) != b"GGUF":
            raise AutotuneError("Expected little-endian GGUF")
        version = r.number("I")
        if version not in (2, 3):
            raise AutotuneError(f"Unsupported GGUF version {version}; supported: 2, 3")
        tensors, fields = r.number("Q"), r.number("Q")
        if tensors > 2_000_000 or fields > 100_000:
            raise AutotuneError("GGUF header exceeds safety limits")
        meta = {}
        for _ in range(fields):
            key = r.string()
            meta[key] = r.value(r.number("I"), not key.startswith("tokenizer."))
        types, layers = Counter(), set()
        for _ in range(tensors):
            name, dims = r.string(), r.number("I")
            if dims > 8:
                raise AutotuneError("Malformed GGUF tensor dimensions")
            shape = [r.number("Q") for _ in range(dims)]
            kind, offset = r.number("I"), r.number("Q")
            if offset > r.size or any(d == 0 for d in shape):
                raise AutotuneError("Malformed GGUF tensor descriptor")
            types[str(kind)] += 1
            match = re.search(r"(?:^|\.)blk\.(\d+)\.", name)
            if match:
                layers.add(int(match.group(1)))
        arch = meta.get("general.architecture")
        return {
            "gguf_version": version,
            "architecture": arch,
            "file_type": meta.get("general.file_type"),
            "quantization_version": meta.get("general.quantization_version"),
            "tensor_count": tensors,
            "tensor_types": dict(types),
            "layers": meta.get(f"{arch}.block_count", len(layers) or None),
            "metadata": meta,
        }


def stat_signature(path: Path) -> dict:
    s = path.stat()
    return {
        "path": str(path),
        "size": s.st_size,
        "mtime_ns": s.st_mtime_ns,
        "ctime_ns": s.st_ctime_ns,
        "device": s.st_dev,
        "inode": s.st_ino,
    }


def fingerprint_file(path: Path, store: Store, rehash: bool = False) -> dict:
    before = stat_signature(path)
    key = digest(str(path))
    old = store.read("models", key)
    if not rehash and old and old.get("stat") == before:
        return old
    info = metadata(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        while data := f.read(8 * 1024 * 1024):
            h.update(data)
    if stat_signature(path) != before:
        raise AutotuneError(
            "Model changed while hashing; retry after the writer finishes"
        )
    result = {"stat": before, "sha256": h.hexdigest(), **info}
    store.write("models", key, result)
    return result


def inspect(path: Path, store: Store, rehash: bool = False) -> dict:
    path = path.expanduser().resolve(strict=True)
    first = fingerprint_file(path, store, rehash)
    count = first["metadata"].get("split.count", 1)
    if not isinstance(count, int) or not 1 <= count <= 10000:
        raise AutotuneError("Invalid GGUF shard count")
    paths = [path]
    if count > 1:
        match = re.fullmatch(r"(.+)-(\d{5})-of-(\d{5})\.gguf", path.name, re.IGNORECASE)
        if not match or int(match[2]) != 1 or int(match[3]) != count:
            raise AutotuneError(
                "Supply the first shard with standard -00001-of-NNNNN.gguf naming"
            )
        paths = [
            path.with_name(f"{match[1]}-{n:05d}-of-{count:05d}.gguf")
            for n in range(1, count + 1)
        ]
    parts = [first] + [fingerprint_file(p, store, rehash) for p in paths[1:]]
    return {
        "path": str(path),
        "size": sum(p["stat"]["size"] for p in parts),
        "fingerprint": digest([p["sha256"] for p in parts]),
        "shards": parts,
        "architecture": first["architecture"],
        "layers": first["layers"],
        "quantization": {
            "file_type": first["file_type"],
            "tensor_types": first["tensor_types"],
        },
        "metadata": first["metadata"],
    }
