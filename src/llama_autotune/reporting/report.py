from pathlib import Path
import re
from ..config import AutotuneError, digest
from ..storage.store import Store


def report(store: Store, model: Path, scope: str | None = None) -> dict:
    if scope:
        if not re.fullmatch(r"[0-9a-f]{64}", scope):
            raise AutotuneError("Scope must be a 64-character hexadecimal fingerprint")
        manifest = store.read("profiles", scope + "-manifest")
        if not manifest:
            raise AutotuneError("No manifest for that scope")
        return {
            "profile": store.read("profiles", scope),
            "manifest": manifest,
            "history": store.history(scope),
        }
    profile = store.read(
        "profiles", digest(str(model.expanduser().resolve())) + "-latest"
    )
    if not profile:
        raise AutotuneError(
            "No successful tuning profile for this model; use --scope to inspect a failed search"
        )
    return {
        "profile": profile,
        "manifest": store.read("profiles", profile["scope"] + "-manifest"),
        "history": store.history(profile["scope"]),
    }
