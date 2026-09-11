from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from .config import AutotuneError


@dataclass
class Result:
    argv: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False
    error: str | None = None
    log: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and self.error is None


@dataclass
class Process:
    process: subprocess.Popen
    stdout_path: Path
    stderr_path: Path
    argv: list[str]

    def output(self) -> tuple[str, str]:
        def tail(path: Path) -> str:
            with path.open("rb") as f:
                f.seek(max(0, path.stat().st_size - 2_000_000))
                return f.read().decode("utf-8", "replace")

        return tail(self.stdout_path), tail(self.stderr_path)


class Executor:
    def __init__(self, logs: Path, verbose: bool = False, debug: bool = False):
        self.logs = logs
        logs.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.debug = debug
        self.current_stage = "running"
        self.gpu_snapshot = "GPU telemetry pending"

    def progress(self, message: str) -> None:
        if self.verbose:
            print(message, file=sys.stderr, flush=True)

    def stop(self, p: subprocess.Popen, grace: float = 5) -> None:
        previous = {}
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, signal.SIG_IGN)
        try:
            self._stop(p, grace)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def _stop(self, p: subprocess.Popen, grace: float) -> None:
        if os.name == "posix":
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        elif p.poll() is None:
            p.terminate()
        try:
            p.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
        if os.name == "posix":
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif p.poll() is None:
            p.kill()
        p.wait(timeout=5)

    @contextmanager
    def managed(
        self, argv: Sequence[str | Path], cwd: Path | None = None, quiet: bool = False
    ) -> Iterator[Process]:
        args = [str(x) for x in argv]
        if self.debug and not quiet:
            print(json.dumps(args), file=sys.stderr, flush=True)
        token = uuid.uuid4().hex
        out, err = self.logs / (token + ".stdout"), self.logs / (token + ".stderr")
        start, p = time.time(), None
        with out.open("wb") as fo, err.open("wb") as fe:
            try:
                env = {
                    k: v
                    for k, v in os.environ.items()
                    if not k.startswith("LLAMA_ARG_")
                }
                env["LC_ALL"] = "C"
                try:
                    p = subprocess.Popen(
                        args,
                        cwd=cwd,
                        stdout=fo,
                        stderr=fe,
                        stdin=subprocess.DEVNULL,
                        start_new_session=os.name == "posix",
                        env=env,
                    )
                except OSError as exc:
                    raise AutotuneError(f"Cannot execute {args[0]}: {exc}") from exc
                heartbeat_done = threading.Event()
                heartbeat = None
                if self.verbose and not quiet:
                    def report_heartbeat():
                        while not heartbeat_done.wait(15):
                            self.progress(f"{self.current_stage} elapsed={time.time()-start:.0f}s {self.gpu_snapshot} heartbeat")
                    heartbeat = threading.Thread(target=report_heartbeat, daemon=True)
                    heartbeat.start()
                try:
                    yield Process(p, out, err, args)
                finally:
                    heartbeat_done.set()
                    if heartbeat:
                        heartbeat.join(timeout=1)
            finally:
                if p is not None:
                    self.stop(p)
                (self.logs / (token + ".json")).write_text(
                    json.dumps(
                        {
                            "argv": args,
                            "cwd": str(cwd) if cwd else None,
                            "started": start,
                            "finished": time.time(),
                            "returncode": p.returncode if p else None,
                            "stdout": str(out),
                            "stderr": str(err),
                        },
                        indent=2,
                    )
                )

    def wait(self, child: Process, timeout: float) -> Result:
        start = time.monotonic()
        timed_out = False
        try:
            child.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            self.stop(child.process)
        out, err = child.output()
        return Result(
            child.argv,
            child.process.returncode,
            out,
            err,
            time.monotonic() - start,
            timed_out,
            log=str(child.stderr_path),
        )

    def run(
        self,
        argv: Sequence[str | Path],
        timeout: float = 20,
        cwd: Path | None = None,
        check: bool = False,
        quiet: bool = False,
    ) -> Result:
        start = time.monotonic()
        try:
            with self.managed(argv, cwd, quiet=quiet) as child:
                result = self.wait(child, timeout)
                result.duration = time.monotonic() - start
        except AutotuneError as exc:
            result = Result(
                [str(x) for x in argv],
                None,
                "",
                "",
                time.monotonic() - start,
                error=str(exc),
            )
        if check and not result.ok:
            raise AutotuneError(
                f"Command failed: {result.argv}; timeout={result.timed_out}; "
                f"{result.error or result.stderr[-2000:]} (log: {result.log})"
            )
        return result
