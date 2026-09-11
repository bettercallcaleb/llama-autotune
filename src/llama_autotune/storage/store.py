from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
import json
import os
import sqlite3
import time
import uuid
from ..config import AutotuneError, SCHEMA, digest


class Store:
    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        for name in ("environment", "builds", "models", "profiles", "logs"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS results (id INTEGER PRIMARY KEY, scope TEXT, candidate TEXT, stage TEXT, created REAL, payload TEXT)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS result_lookup ON results(scope,candidate,stage,created)"
            )

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.root / "results.sqlite", timeout=30)
        try:
            db.execute("PRAGMA journal_mode=WAL")
            with db:
                yield db
        finally:
            db.close()

    def write(self, category: str, key: str, value: dict) -> Path:
        path = self.root / category / (key + ".json")
        temp = path.with_suffix("." + uuid.uuid4().hex + ".tmp")
        temp.write_text(
            json.dumps({"schema": SCHEMA, **value}, indent=2, sort_keys=True)
        )
        os.replace(temp, path)
        return path

    def read(self, category: str, key: str) -> dict | None:
        try:
            value = json.loads((self.root / category / (key + ".json")).read_text())
            return (
                value
                if isinstance(value, dict) and value.get("schema") == SCHEMA
                else None
            )
        except (ValueError, OSError):
            return None

    def add(self, scope: str, config: dict, stage: str, result: dict) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO results(scope,candidate,stage,created,payload) VALUES (?,?,?,?,?)",
                (
                    scope,
                    digest(config),
                    stage,
                    time.time(),
                    json.dumps({"config": config, **result}),
                ),
            )

    def previous(self, scope: str, config: dict, stage: str) -> dict | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT payload FROM results WHERE scope=? AND candidate=? AND stage=? ORDER BY id DESC LIMIT 1",
                (scope, digest(config), stage),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def history(self, scope: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT stage,created,payload FROM results WHERE scope=? ORDER BY id",
                (scope,),
            ).fetchall()
        return [{"stage": s, "created": t, **json.loads(p)} for s, t, p in rows]

    @contextmanager
    def lock(self) -> Iterator[None]:
        if os.name != "posix":
            raise AutotuneError(
                "v0 build/tune/run requires Linux; probe and inspect are portable"
            )
        import fcntl

        with (self.root / "execution.lock").open("a") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise AutotuneError(
                    "Another build/tune/run owns this cache. Wait for it to finish."
                ) from exc
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
