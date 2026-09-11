from dataclasses import asdict, dataclass
import math
from ..config import AutotuneError, RuntimeConfig, Workload, digest


@dataclass(frozen=True)
class DepthPlan:
    requested_ctx: int
    per_slot_ctx: int
    requested_depth_policy: str
    depth_ratio: float
    effective_validation_depth: int
    coarse_prompt_tokens: int
    generation_token_count: int
    boundary_reserve: int = 32

    def json(self) -> dict:
        return asdict(self)


def resolve_depth(config: RuntimeConfig, workload: Workload) -> DepthPlan:
    if config.ctx < 1 or config.parallel < 1 or workload.tokens < 1:
        raise AutotuneError(
            "Context, parallelism and generation count must be positive"
        )
    if not math.isfinite(workload.depth_ratio) or not 0 < workload.depth_ratio < 1:
        raise AutotuneError(
            "--depth-ratio must be finite and between 0 and 1 (exclusive)"
        )
    per_slot = config.ctx // config.parallel
    maximum = per_slot - workload.tokens - 32 - 1
    if maximum < 1:
        raise AutotuneError(
            "Per-slot context must fit depth + generation + 32 safety tokens + 1 benchmark token"
        )
    if workload.depth == "auto":
        depth = min(maximum, max(1, int(per_slot * workload.depth_ratio)))
    else:
        try:
            depth = int(workload.depth)
        except (ValueError, TypeError) as exc:
            raise AutotuneError("--depth must be auto or a positive integer") from exc
        if not 1 <= depth <= maximum:
            raise AutotuneError(
                f"--depth must be between 1 and {maximum} for this per-slot context and generation count"
            )
    pp = min(512, per_slot - depth - workload.tokens - 32)
    return DepthPlan(
        config.ctx,
        per_slot,
        str(workload.depth),
        workload.depth_ratio,
        depth,
        pp,
        workload.tokens,
    )


def token_ids(response: dict) -> list[int]:
    values = response.get("tokens")
    if not isinstance(values, list) or not values:
        raise AutotuneError(
            "Tokenizer did not return nonempty token IDs; depth validation is unavailable"
        )
    if any(type(t) is not int or t < 0 for t in values):
        raise AutotuneError("Malformed tokenizer token IDs")
    return values


def depth_prompt(
    request, workload: Workload, plan: DepthPlan
) -> tuple[list[int], dict]:
    tail = token_ids(
        request(
            "/tokenize",
            {"content": workload.prompt, "add_special": False, "parse_special": False},
            timeout=workload.request_timeout,
        )
    )
    filler = token_ids(
        request(
            "/tokenize",
            {
                "content": " A quiet river passes through the valley. Trees grow beside the water. This is deterministic context for a performance measurement.",
                "add_special": False,
                "parse_special": False,
            },
            timeout=workload.request_timeout,
        )
    )
    depth = plan.effective_validation_depth
    suffix = tail[-min(len(tail), depth) :]
    needed = depth - len(suffix)
    tokens = (filler * ((needed + len(filler) - 1) // len(filler)))[:needed] + suffix
    return tokens, {
        **plan.json(),
        "prompt_token_count": len(tokens),
        "prompt_token_sha256": digest(tokens),
        "filler_token_count": needed,
        "workload_suffix_token_count": len(suffix),
        "workload_suffix_truncated": len(tail) > len(suffix),
        "filler_generated_by_model": False,
    }
