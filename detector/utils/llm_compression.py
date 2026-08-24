#!/usr/bin/env python3
"""
Shared LLM-only compression helpers for detector prompts.
"""

import asyncio
import hashlib
import json
import logging
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =====================================================================
# Three-tier per-step compression (shared by Phase 1 / 1.5 / 2)
# ---------------------------------------------------------------------
# Every step in a trajectory is precomputed once at three different
# output budgets (th1 > th2 > th3) and the three results are stored on
# the step. Downstream stages then pick the right tier per step based
# on the step's role relative to "current" (Phase 1) or "error step"
# (Phase 1.5 / Phase 2). No stage re-compresses.
#
# Consumer rules (see callers for exact mapping):
#   th1: Phase 1 current step;
#        Phase 1.5 origin error step + the immediate prev step;
#        Phase 2 steps that have a Phase-1 detected error.
#   th2: Phase 1 +/-2 neighbors of current;
#        Phase 1.5 forward window snippets;
#        Phase 2 +/-2 neighbors of any error step.
#   th3: Phase 1 everything farther than +/-2;
#        Phase 2 steps with no error and not adjacent to any error step.
#
# Per-step output length cap at each tier (smaller = more compressed):
DEFAULT_STEP_TH1_MAX_CHARS = 1024
DEFAULT_STEP_TH2_MAX_CHARS = 512
DEFAULT_STEP_TH3_MAX_CHARS = 256

# Default max_tokens for the unified (single-call-per-step) compression.
# This is the LLM API max_tokens cap on the *output* side. Set high
# because reasoning models emit thinking tokens far exceeding the
# compressed output size; callers override via --stage_a_max_tokens.
DEFAULT_STAGE_A_MAX_TOKENS = 32768

# Steps with raw content <= the corresponding tier max are stored
# verbatim (short-circuit). If every step in a chunk short-circuits,
# that chunk skips the LLM call entirely.

# Chunk sizes for the batched precomputation LLM calls. One LLM call
# per chunk; chunks within a tier run in parallel via asyncio.gather.
DEFAULT_TH1_CHUNK_SIZE = 1
DEFAULT_TH2_CHUNK_SIZE = 3
DEFAULT_TH3_CHUNK_SIZE = 5

# Hard safety cap on the total rendered history block (Task + per-step
# tiered renderings). Applied as a final clip_text_middle in consumers.
DEFAULT_COMPRESSED_HISTORY_OVERALL_CAP_CHARS = 12000

# ---------------------------------------------------------------------
# Auto-estimation for per-tier compression ``max_tokens``.
#
# ``_compress_chunk_async`` asks the LLM to emit a JSON payload of the
# form
#     {"steps": [{"step": N, "agent": "<=max_chars>", "env": "<=max_chars>"}, ...]}
# so the worst-case output length is roughly
#     2 * max_chars * chunk_size         (agent + env text per step)
#   +   ~60 chars                        (JSON keys / braces / quotes)
#     per step                           (escapes for newlines, quotes, etc.)
#
# The previous hardcoded ``max_tokens=1024`` would silently truncate the
# response for th1 (max_chars=1500 even with chunk_size=1) and th2 with
# chunk_size>=3, after which ``_parse_chunk_compress_response`` failed
# and the tier fell back to a deterministic ``clip_text_middle``. That
# degraded Phase 1 quality invisibly.
#
# We now derive ``max_tokens`` from ``max_chars * chunk_size`` with a
# conservative chars-per-token ratio (1.3 covers Chinese + JSON escape
# overhead), plus a small constant buffer for the envelope, and a
# generous floor so trivially small tiers still have headroom. Callers
# can still pass an explicit ``max_tokens`` to pin the budget when a
# specific backend has different tokenization behavior.
COMPRESS_MAX_TOKENS_CHARS_PER_TOKEN = 1.3
COMPRESS_MAX_TOKENS_ENVELOPE_CHARS = 400
COMPRESS_MAX_TOKENS_TAIL_BUFFER = 128
COMPRESS_MAX_TOKENS_FLOOR = 1024

# Backend context-length cap (input + output) in **tokens**, matching
# sglang / vLLM ``--context-length`` / ``max_total_tokens``. Used by
# ``_compress_chunk_async`` to adaptively split chunks (or, as a last
# resort, clip a single-step chunk's raw content) so that the rendered
# prompt + requested output never blows past the model's hard window.
# Callers can override via ``aprecompute_step_compressions(max_lm_tokens=...)``.
DEFAULT_COMPRESS_MAX_LM_TOKENS = 131072

# Rough fixed-overhead (chars) of the ``_build_chunk_compress_prompt``
# template itself (instructions, JSON schema echo, preserve-verbatim
# block), excluding the per-step payload. Measured at ~2.3KB on the
# current template; rounded up to 3500 for safety headroom.
COMPRESS_PROMPT_TEMPLATE_OVERHEAD_CHARS = 3500


def estimate_compress_max_tokens(max_chars: int, chunk_size: int) -> int:
    """Auto-estimate ``max_tokens`` for one tier-compress LLM call.

    ``max_chars`` is the per-side (Agent or Env) char cap; output
    contains both sides per step, hence the ``2 *`` factor.
    """
    total_chars = (
        2 * max(1, int(max_chars)) * max(1, int(chunk_size))
        + COMPRESS_MAX_TOKENS_ENVELOPE_CHARS
    )
    estimate = int(total_chars / COMPRESS_MAX_TOKENS_CHARS_PER_TOKEN) + COMPRESS_MAX_TOKENS_TAIL_BUFFER
    return max(COMPRESS_MAX_TOKENS_FLOOR, estimate)


def _estimate_chunk_input_tokens(chunk: List[Dict[str, Any]]) -> int:
    """Approximate input-side token count for a chunk rendered through
    :func:`_build_chunk_compress_prompt`.

    Counts: template fixed overhead + per-entry header (~48 chars for
    ``[step=N role=...]\n``) + the raw ``content`` length, then divides
    by :data:`COMPRESS_MAX_TOKENS_CHARS_PER_TOKEN` to convert to tokens.
    Used only for budgeting; not a replacement for the real tokenizer.
    """
    total_chars = COMPRESS_PROMPT_TEMPLATE_OVERHEAD_CHARS
    for entry in chunk:
        total_chars += 48  # per-entry header + separators
        total_chars += len(str(entry.get("content", "") or ""))
    return int(total_chars / COMPRESS_MAX_TOKENS_CHARS_PER_TOKEN) + 1

# =====================================================================
# Legacy constants (retained as aliases for backward-compat fallbacks
# in Phase 1.5 / 2 when reading older Phase-1 JSON outputs that lack
# the `step_compressions` field, and as defaults for the deprecated
# single-message LLM-compress helpers further down).
# =====================================================================
DEFAULT_HISTORY_COMPRESS_THRESHOLD_CHARS = 8192
DEFAULT_ENV_COMPRESS_THRESHOLD_CHARS = 1200
DEFAULT_HISTORY_COMPRESS_MAX_CHARS = 4096
DEFAULT_ENV_COMPRESS_MAX_CHARS = 900
DEFAULT_HISTORY_BLOCK_SIZE = 3
DEFAULT_HISTORY_RECENT_RAW_STEPS = 2
DEFAULT_HISTORY_BLOCK_SUMMARY_MAX_CHARS = 320
DEFAULT_AGENT_COGNITION_MAX_CHARS = 1800


def _squeeze_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


# Role tokens that strongly indicate an "agent cognition" message
# (planning, reasoning, delegation, thought). These messages must NOT be
# aggressively compressed in history, so that downstream modules can still
# see the authoritative plan/thought when judging PLAN/REASON errors.
_AGENT_ROLE_TOKENS = (
    "orchestrator",
    "planner",
    "assistant_thought",
    "thought",
    "system_plan",
)

# Role tokens that indicate an "environment feedback" or executor message
# (tool output, page dump, OCR, screenshot, terminal output). These are
# safe to compress down to key entities.
_ENV_ROLE_TOKENS = (
    "websurfer",
    "filesurfer",
    "computerterminal",
    "tool",
    "observation",
    "env",
    "user",  # in multi-agent traces user messages often carry env-like content
)

_AGENT_COGNITION_MARKERS = (
    "Initial plan",
    "Here is the plan",
    "Updated Ledger",
    "Next speaker",
    "Please search",
    "Please click",
    "plan",
)


def classify_message_kind(
    role: str,
    content: str = "",
    env_response: Optional[str] = None,
) -> str:
    """Classify a history message as agent cognition vs environment feedback.

    Returns one of ``"agent_cognition"``, ``"env_feedback"``, ``"mixed"``.

    Priority:
      1) If ``env_response`` is explicitly provided (structured envs already
         split agent vs env), the caller can treat the agent ``content`` as
         ``agent_cognition`` and the ``env_response`` as ``env_feedback``
         directly; in that case callers typically invoke this helper twice
         with the respective payloads and an empty ``env_response``.
      2) Fall back to role-token matching on ``role``.
      3) If role does not match anything known, use content heuristics
         (plan/ledger markers vs screenshot/OCR markers).
      4) Otherwise return ``"mixed"``.
    """

    role_lc = str(role or "").strip().lower()
    for token in _AGENT_ROLE_TOKENS:
        if token in role_lc:
            return "agent_cognition"
    for token in _ENV_ROLE_TOKENS:
        if token in role_lc:
            return "env_feedback"

    text = str(content or "")
    if any(marker in text for marker in _AGENT_COGNITION_MARKERS):
        return "agent_cognition"
    if (
        "screenshot" in text.lower()
        or "viewport" in text.lower()
        or "automatic ocr" in text.lower()
        or "tool_call:" in text.lower()
    ):
        return "env_feedback"

    return "mixed"


def is_initial_plan_message(role: str, content: str) -> bool:
    """Detect the authoritative initial plan message that must stay sticky
    at the head of the compressed history.

    Heuristic: role is agent cognition AND content mentions an explicit
    plan / fact-sheet / initial-plan marker. Conservative on purpose so we
    only ever pin one or two messages.
    """

    if classify_message_kind(role, content) != "agent_cognition":
        return False
    text = str(content or "")
    markers = (
        "Initial plan",
        "Here is the plan",
        "Here is an initial fact sheet",
        "Here is the plan to follow",
    )
    return any(m in text for m in markers)


def clip_text_middle(text: str, limit: int) -> str:
    source = str(text or "")
    if len(source) <= limit:
        return source
    if limit <= 40:
        return source[:limit]
    head = int(limit * 0.72)
    tail = max(0, limit - head - 16)
    return source[:head] + "\n...[compressed]...\n" + (source[-tail:] if tail else "")


def _extract_json_object(payload: str) -> str:
    start = payload.find("{")
    if start < 0:
        return ""
    depth = 0
    for idx in range(start, len(payload)):
        ch = payload[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return payload[start: idx + 1]
    return ""


_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_DANGLING_THINK_OPEN_RE = re.compile(r"<think\b[^>]*>.*?$", re.IGNORECASE | re.DOTALL)
_ORPHAN_THINK_CLOSE_RE = re.compile(r"^.*?</think\s*>", re.IGNORECASE | re.DOTALL)


def strip_think_tags(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks emitted by reasoning
    models (DeepSeek-R1, Qwen QwQ, etc.) before JSON extraction.

    Tolerates three malformed shapes that reasoning backends occasionally
    leak through when ``response_format={"type": "json_object"}`` is NOT
    enforced:

    * Properly paired ``<think>...</think>`` blocks anywhere in the text.
    * A dangling unclosed ``<think>...`` prefix (model ran out of tokens
      before writing the closing tag).
    * An orphan ``</think>`` with no opener (prompt-side scaffolding that
      bled into the completion).

    Pure string transform; safe to call even when ``force_json=True`` so
    callers never need to branch on the flag before parsing.
    """
    s = str(text or "")
    s = _THINK_BLOCK_RE.sub("", s)
    if "<think" in s.lower() and "</think" not in s.lower():
        s = _DANGLING_THINK_OPEN_RE.sub("", s)
    elif "</think" in s.lower() and "<think" not in s.lower():
        s = _ORPHAN_THINK_CLOSE_RE.sub("", s)
    return s.strip()


def _parse_json_text_field(payload: str) -> str:
    try:
        obj = json.loads(payload)
    except Exception:
        return ""
    if not isinstance(obj, dict):
        return ""
    for key in ("compressed_text", "summary", "result", "text"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _parse_compression_response(response_text: str) -> str:
    payload = strip_think_tags(str(response_text or "")).strip()
    if not payload:
        return ""

    parsed = _parse_json_text_field(payload)
    if parsed:
        return parsed

    json_obj = _extract_json_object(payload)
    if json_obj:
        parsed = _parse_json_text_field(json_obj)
        if parsed:
            return parsed

    # Fallback: strip code fences and return raw text.
    no_fence = re.sub(r"^```(?:json)?\s*|\s*```$", "", payload, flags=re.IGNORECASE | re.DOTALL).strip()
    return no_fence


def build_compression_prompt(
    content_kind: str,
    source_text: str,
    max_output_chars: int,
) -> str:
    kind = "history input" if content_kind == "history" else "environment feedback"
    focus = (
        "Preserve chronology and any concrete actions/observations/errors."
        if content_kind == "history"
        else "Preserve current-step outcomes, observations, and explicit errors."
    )
    return f"""
Compress the following {kind} for downstream error analysis.
Do not invent facts. Keep concrete strings like tool names, URLs, numbers, and error messages.
Preserve agent summaries, memories, reflections, progress/completion claims, and the concrete evidence they cite.
{focus}

Return ONLY one JSON object:
{{
  "compressed_text": "..."
}}

Constraints:
- Output length <= {max_output_chars} characters.
- Keep language concise and factual.

SOURCE TEXT:
{source_text}
"""


def _append_history_record(history_input: str, agent_output: str, env_response: str) -> str:
    blocks: List[str] = []
    if history_input and history_input.strip():
        blocks.append(history_input.strip())
    blocks.append("[Previous Step Agent Thought & Action]\n" + (agent_output or "").strip())
    blocks.append("[Previous Step Environment Feedback]\n" + (env_response or "").strip())
    return "\n\n".join(blocks).strip()


def build_history_block_summary_prompt(
    task_description: str,
    block_start: int,
    block_end: int,
    block_payload: str,
    max_output_chars: int,
    summary_kind: str = "default",
) -> str:
    return f"""
Summarize this trajectory history block for downstream step-level error diagnosis.
Do not invent facts.
Keep concrete entities (tool names, URLs, numbers, error strings), and preserve step numbering.
Preserve agent summaries, memories, reflections, progress/completion claims, and the evidence they cite.

TASK:
{task_description}

BLOCK RANGE:
steps {block_start} to {block_end}

BLOCK CONTENT:
{block_payload}

Return ONLY one JSON object:
{{
  "summary": "..."
}}

Constraints:
- Output length <= {max_output_chars} characters.
- Focus on factual actions, key observations, explicit errors, and unresolved state.
"""


def summarize_history_block_with_llm(
    generate_fn: Callable[..., Any],
    task_description: str,
    block_start: int,
    block_end: int,
    block_payload: str,
    max_output_chars: int,
    summary_kind: str = "default",
    max_tokens: int = 512,
    force_json: bool = True,
) -> str:
    prompt = build_history_block_summary_prompt(
        task_description=task_description,
        block_start=block_start,
        block_end=block_end,
        block_payload=block_payload,
        max_output_chars=max_output_chars,
        summary_kind=summary_kind,
    )

    summary = ""
    try:
        gen_kwargs: Dict[str, Any] = {"max_tokens": max_tokens, "temperature": 0.0}
        if force_json:
            gen_kwargs["response_format"] = {"type": "json_object"}
        response = generate_fn(prompt, **gen_kwargs)
        summary = _parse_compression_response(str(response or ""))
    except Exception:
        summary = ""

    if not summary:
        summary = clip_text_middle(block_payload, max_output_chars)
    elif len(summary) > max_output_chars:
        summary = clip_text_middle(summary, max_output_chars)

    summary = _squeeze_whitespace(summary)
    if not summary:
        summary = clip_text_middle(block_payload, max_output_chars)

    return summary


def compose_history_with_block_summaries(
    generate_fn: Callable[..., Any],
    task_id: str,
    task_description: str,
    prior_steps: List[Dict[str, Any]],
    threshold_chars: int,
    block_size: int = DEFAULT_HISTORY_BLOCK_SIZE,
    recent_raw_steps: int = DEFAULT_HISTORY_RECENT_RAW_STEPS,
    block_summary_max_chars: int = DEFAULT_HISTORY_BLOCK_SUMMARY_MAX_CHARS,
    cache: Optional[Dict[str, str]] = None,
    summary_kind: str = "default",
    agent_cognition_max_chars: int = DEFAULT_AGENT_COGNITION_MAX_CHARS,
    overall_cap_chars: int = DEFAULT_COMPRESSED_HISTORY_OVERALL_CAP_CHARS,
    force_json: bool = True,
) -> Tuple[str, bool]:
    """Role-aware structured-history composer.

    Each ``prior_step`` is a pair of (agent_text, env_text). We treat the
    agent side as planning/reasoning evidence and preserve it raw (with
    per-message clipping), while env_text of older steps is grouped into
    blocks and summarized via the existing LLM block summarizer. The
    first step's agent_text is pinned as a sticky initial plan if it
    matches plan markers.
    """
    task_text = str(task_description or "").strip() or "Start of task"
    if not prior_steps:
        return task_text, False

    raw_history = task_text
    for step in prior_steps:
        agent_text = str(step.get("history_content", step.get("content", step.get("agent_response", ""))))
        env_text = str(step.get("history_env_response", step.get("env_response", "")))
        raw_history = _append_history_record(raw_history, agent_text, env_text)

    if len(raw_history) <= max(0, int(threshold_chars)):
        return raw_history, False

    keep_n = max(1, int(recent_raw_steps))
    if len(prior_steps) <= keep_n:
        return raw_history, False

    if cache is None:
        cache = {}

    older_steps = prior_steps[:-keep_n]
    recent_steps = prior_steps[-keep_n:]

    # Sticky initial plan: first older step whose agent_text matches plan markers.
    sticky_idx: Optional[int] = None
    for idx, step in enumerate(older_steps):
        agent_text = str(step.get("history_content", step.get("content", step.get("agent_response", ""))))
        # Role is usually not encoded on structured steps — treat as orchestrator.
        if is_initial_plan_message("orchestrator", agent_text):
            sticky_idx = idx
            break

    group_size = max(1, int(block_size))
    env_summaries: List[Tuple[int, int, str]] = []
    preserved_agent_lines: List[str] = []

    for i in range(0, len(older_steps), group_size):
        block = older_steps[i:i + group_size]
        if not block:
            continue

        try:
            block_start = int(block[0].get("step", i + 1))
        except Exception:
            block_start = i + 1
        try:
            block_end = int(block[-1].get("step", block_start + len(block) - 1))
        except Exception:
            block_end = block_start + len(block) - 1

        # Preserve agent_text per step (skip sticky one, emitted separately).
        for local_idx, entry in enumerate(block):
            global_idx = i + local_idx
            if global_idx == sticky_idx:
                continue
            step_num = entry.get("step", "")
            agent_text = str(entry.get("history_content", entry.get("content", entry.get("agent_response", ""))))
            if agent_text.strip():
                clipped = clip_text_middle(agent_text, max(200, int(agent_cognition_max_chars)))
                preserved_agent_lines.append(f"[Step {step_num} Agent]\n{clipped}")

        # Summarize env_text across the block.
        env_lines: List[str] = []
        for entry in block:
            step_num = entry.get("step", "")
            env_text = str(entry.get("history_env_response", entry.get("env_response", "")))
            if env_text.strip():
                env_lines.append(f"[Step {step_num} Env]\n{env_text}")

        if not env_lines:
            continue

        block_payload = "\n\n".join(env_lines).strip()
        digest = hashlib.sha1(block_payload.encode("utf-8")).hexdigest()[:16]
        cache_key = f"{task_id}|env|{block_start}-{block_end}|{digest}"
        summary_text = cache.get(cache_key, "")
        if not summary_text:
            summary_text = summarize_history_block_with_llm(
                generate_fn=generate_fn,
                task_description=task_text,
                block_start=block_start,
                block_end=block_end,
                block_payload=block_payload,
                max_output_chars=block_summary_max_chars,
                summary_kind=summary_kind,
                force_json=force_json,
            )
            cache[cache_key] = summary_text
        env_summaries.append((block_start, block_end, summary_text))

    parts: List[str] = []
    parts.append("[Task]\n" + task_text)

    if sticky_idx is not None:
        sticky_step = older_steps[sticky_idx]
        step_num = sticky_step.get("step", "")
        agent_text = str(sticky_step.get("history_content", sticky_step.get("content", sticky_step.get("agent_response", ""))))
        clipped = clip_text_middle(agent_text, max(400, int(agent_cognition_max_chars)))
        parts.append(f"[Sticky Initial Plan]\n[Step {step_num} Agent]\n{clipped}")

    if env_summaries:
        lines = ["[Earlier Env Summaries]"]
        for start, end, summary in env_summaries:
            label = f"Steps {start}-{end}" if start != end else f"Step {start}"
            lines.append(f"- {label}: {summary}")
        parts.append("\n".join(lines))

    if preserved_agent_lines:
        parts.append("\n\n".join(["[Earlier Agent Cognition Raw]"] + preserved_agent_lines))

    recent_lines: List[str] = ["[Recent Steps Raw]"]
    for entry in recent_steps:
        step_num = entry.get("step", "")
        agent_text = str(entry.get("history_content", entry.get("content", entry.get("agent_response", ""))))
        env_text = str(entry.get("history_env_response", entry.get("env_response", "")))
        recent_lines.append(f"[Step {step_num} Agent]\n{agent_text}")
        recent_lines.append(f"[Step {step_num} Env]\n{env_text}")
    parts.append("\n\n".join(recent_lines))

    rendered = "\n\n".join(parts).strip()
    if not rendered:
        return raw_history, False

    cap = max(0, int(overall_cap_chars))
    if cap and len(rendered) > cap:
        rendered = clip_text_middle(rendered, cap)

    return rendered, True


def compress_text_with_llm(
    generate_fn: Callable[..., Any],
    text: str,
    content_kind: str,
    threshold_chars: int,
    max_output_chars: int,
    max_tokens: int = 1024,
    force_json: bool = True,
) -> Tuple[str, bool]:
    source = str(text or "")
    if not source.strip():
        return source, False
    if len(source) <= max(0, int(threshold_chars)):
        return source, False

    prompt = build_compression_prompt(
        content_kind=content_kind,
        source_text=source,
        max_output_chars=max_output_chars,
    )

    compressed = ""
    try:
        gen_kwargs: Dict[str, Any] = {"max_tokens": max_tokens, "temperature": 0.0}
        if force_json:
            gen_kwargs["response_format"] = {"type": "json_object"}
        response = generate_fn(prompt, **gen_kwargs)
        compressed = _parse_compression_response(str(response or ""))
    except Exception:
        compressed = ""

    if not compressed:
        compressed = clip_text_middle(source, max_output_chars)
    elif len(compressed) > max_output_chars:
        compressed = clip_text_middle(compressed, max_output_chars)

    compressed = _squeeze_whitespace(compressed)
    if not compressed:
        compressed = clip_text_middle(source, max_output_chars)

    return compressed, True


# ----------------------------------------------------------------------
# Async-friendly siblings of the helpers above.
#
# These accept an *async* ``agenerate_fn`` (e.g. one that internally does
# ``await asyncio.to_thread(model.generate, ...)``) so callers running
# inside an asyncio event loop don't block the loop while compressing.
# Behavior / return types are identical to the sync versions.
# ----------------------------------------------------------------------


async def asummarize_history_block_with_llm(
    agenerate_fn: Callable[..., Awaitable[Any]],
    task_description: str,
    block_start: int,
    block_end: int,
    block_payload: str,
    max_output_chars: int,
    summary_kind: str = "default",
    max_tokens: int = 512,
    force_json: bool = True,
) -> str:
    prompt = build_history_block_summary_prompt(
        task_description=task_description,
        block_start=block_start,
        block_end=block_end,
        block_payload=block_payload,
        max_output_chars=max_output_chars,
        summary_kind=summary_kind,
    )

    summary = ""
    try:
        gen_kwargs: Dict[str, Any] = {"max_tokens": max_tokens, "temperature": 0.0}
        if force_json:
            gen_kwargs["response_format"] = {"type": "json_object"}
        response = await agenerate_fn(prompt, **gen_kwargs)
        summary = _parse_compression_response(str(response or ""))
    except Exception:
        summary = ""

    if not summary:
        summary = clip_text_middle(block_payload, max_output_chars)
    elif len(summary) > max_output_chars:
        summary = clip_text_middle(summary, max_output_chars)

    summary = _squeeze_whitespace(summary)
    if not summary:
        summary = clip_text_middle(block_payload, max_output_chars)

    return summary


async def acompose_history_with_block_summaries(
    agenerate_fn: Callable[..., Awaitable[Any]],
    task_id: str,
    task_description: str,
    prior_steps: List[Dict[str, Any]],
    threshold_chars: int,
    block_size: int = DEFAULT_HISTORY_BLOCK_SIZE,
    recent_raw_steps: int = DEFAULT_HISTORY_RECENT_RAW_STEPS,
    block_summary_max_chars: int = DEFAULT_HISTORY_BLOCK_SUMMARY_MAX_CHARS,
    cache: Optional[Dict[str, str]] = None,
    summary_kind: str = "default",
    agent_cognition_max_chars: int = DEFAULT_AGENT_COGNITION_MAX_CHARS,
    overall_cap_chars: int = DEFAULT_COMPRESSED_HISTORY_OVERALL_CAP_CHARS,
    inflight_cache: Optional[Dict[str, "asyncio.Future[str]"]] = None,
    force_json: bool = True,
) -> Tuple[str, bool]:
    """Async sibling of :func:`compose_history_with_block_summaries`.

    When ``inflight_cache`` is provided, concurrent callers that miss the
    same ``cache_key`` share a single LLM call via an ``asyncio.Future``.
    This is the recommended path when many step-level pipelines are
    composing histories in parallel.
    """
    task_text = str(task_description or "").strip() or "Start of task"
    if not prior_steps:
        return task_text, False

    raw_history = task_text
    for step in prior_steps:
        agent_text = str(step.get("history_content", step.get("content", step.get("agent_response", ""))))
        env_text = str(step.get("history_env_response", step.get("env_response", "")))
        raw_history = _append_history_record(raw_history, agent_text, env_text)

    if len(raw_history) <= max(0, int(threshold_chars)):
        return raw_history, False

    keep_n = max(1, int(recent_raw_steps))
    if len(prior_steps) <= keep_n:
        return raw_history, False

    if cache is None:
        cache = {}

    older_steps = prior_steps[:-keep_n]
    recent_steps = prior_steps[-keep_n:]

    sticky_idx: Optional[int] = None
    for idx, step in enumerate(older_steps):
        agent_text = str(step.get("history_content", step.get("content", step.get("agent_response", ""))))
        if is_initial_plan_message("orchestrator", agent_text):
            sticky_idx = idx
            break

    group_size = max(1, int(block_size))
    env_summaries: List[Tuple[int, int, str]] = []
    preserved_agent_lines: List[str] = []

    for i in range(0, len(older_steps), group_size):
        block = older_steps[i:i + group_size]
        if not block:
            continue

        try:
            block_start = int(block[0].get("step", i + 1))
        except Exception:
            block_start = i + 1
        try:
            block_end = int(block[-1].get("step", block_start + len(block) - 1))
        except Exception:
            block_end = block_start + len(block) - 1

        for local_idx, entry in enumerate(block):
            global_idx = i + local_idx
            if global_idx == sticky_idx:
                continue
            step_num = entry.get("step", "")
            agent_text = str(entry.get("history_content", entry.get("content", entry.get("agent_response", ""))))
            if agent_text.strip():
                clipped = clip_text_middle(agent_text, max(200, int(agent_cognition_max_chars)))
                preserved_agent_lines.append(f"[Step {step_num} Agent]\n{clipped}")

        env_lines: List[str] = []
        for entry in block:
            step_num = entry.get("step", "")
            env_text = str(entry.get("history_env_response", entry.get("env_response", "")))
            if env_text.strip():
                env_lines.append(f"[Step {step_num} Env]\n{env_text}")

        if not env_lines:
            continue

        block_payload = "\n\n".join(env_lines).strip()
        digest = hashlib.sha1(block_payload.encode("utf-8")).hexdigest()[:16]
        cache_key = f"{task_id}|env|{block_start}-{block_end}|{digest}"
        summary_text = cache.get(cache_key, "")
        if not summary_text:
            if inflight_cache is not None:
                fut = inflight_cache.get(cache_key)
                if fut is None:
                    fut = asyncio.get_running_loop().create_future()
                    inflight_cache[cache_key] = fut
                    try:
                        produced = await asummarize_history_block_with_llm(
                            agenerate_fn=agenerate_fn,
                            task_description=task_text,
                            block_start=block_start,
                            block_end=block_end,
                            block_payload=block_payload,
                            max_output_chars=block_summary_max_chars,
                            summary_kind=summary_kind,
                            force_json=force_json,
                        )
                        cache[cache_key] = produced
                        if not fut.done():
                            fut.set_result(produced)
                        summary_text = produced
                    except Exception as exc:
                        if not fut.done():
                            fut.set_exception(exc)
                        inflight_cache.pop(cache_key, None)
                        raise
                    else:
                        inflight_cache.pop(cache_key, None)
                else:
                    summary_text = await fut
            else:
                summary_text = await asummarize_history_block_with_llm(
                    agenerate_fn=agenerate_fn,
                    task_description=task_text,
                    block_start=block_start,
                    block_end=block_end,
                    block_payload=block_payload,
                    max_output_chars=block_summary_max_chars,
                    summary_kind=summary_kind,
                    force_json=force_json,
                )
                cache[cache_key] = summary_text
        env_summaries.append((block_start, block_end, summary_text))

    parts: List[str] = []
    parts.append("[Task]\n" + task_text)

    if sticky_idx is not None:
        sticky_step = older_steps[sticky_idx]
        step_num = sticky_step.get("step", "")
        agent_text = str(sticky_step.get("history_content", sticky_step.get("content", sticky_step.get("agent_response", ""))))
        clipped = clip_text_middle(agent_text, max(400, int(agent_cognition_max_chars)))
        parts.append(f"[Sticky Initial Plan]\n[Step {step_num} Agent]\n{clipped}")

    if env_summaries:
        lines = ["[Earlier Env Summaries]"]
        for start, end, summary in env_summaries:
            label = f"Steps {start}-{end}" if start != end else f"Step {start}"
            lines.append(f"- {label}: {summary}")
        parts.append("\n".join(lines))

    if preserved_agent_lines:
        parts.append("\n\n".join(["[Earlier Agent Cognition Raw]"] + preserved_agent_lines))

    recent_lines: List[str] = ["[Recent Steps Raw]"]
    for entry in recent_steps:
        step_num = entry.get("step", "")
        agent_text = str(entry.get("history_content", entry.get("content", entry.get("agent_response", ""))))
        env_text = str(entry.get("history_env_response", entry.get("env_response", "")))
        recent_lines.append(f"[Step {step_num} Agent]\n{agent_text}")
        recent_lines.append(f"[Step {step_num} Env]\n{env_text}")
    parts.append("\n\n".join(recent_lines))

    rendered = "\n\n".join(parts).strip()
    if not rendered:
        return raw_history, False

    cap = max(0, int(overall_cap_chars))
    if cap and len(rendered) > cap:
        rendered = clip_text_middle(rendered, cap)

    return rendered, True


async def acompress_text_with_llm(
    agenerate_fn: Callable[..., Awaitable[Any]],
    text: str,
    content_kind: str,
    threshold_chars: int,
    max_output_chars: int,
    max_tokens: int = 1024,
    force_json: bool = True,
) -> Tuple[str, bool]:
    """Async sibling of :func:`compress_text_with_llm`."""
    source = str(text or "")
    if not source.strip():
        return source, False
    if len(source) <= max(0, int(threshold_chars)):
        return source, False

    prompt = build_compression_prompt(
        content_kind=content_kind,
        source_text=source,
        max_output_chars=max_output_chars,
    )

    compressed = ""
    try:
        gen_kwargs: Dict[str, Any] = {"max_tokens": max_tokens, "temperature": 0.0}
        if force_json:
            gen_kwargs["response_format"] = {"type": "json_object"}
        response = await agenerate_fn(prompt, **gen_kwargs)
        compressed = _parse_compression_response(str(response or ""))
    except Exception:
        compressed = ""

    if not compressed:
        compressed = clip_text_middle(source, max_output_chars)
    elif len(compressed) > max_output_chars:
        compressed = clip_text_middle(compressed, max_output_chars)

    compressed = _squeeze_whitespace(compressed)
    if not compressed:
        compressed = clip_text_middle(source, max_output_chars)

    return compressed, True


# =====================================================================
# Three-tier per-step precomputation entry point.
# ---------------------------------------------------------------------
# Used by Phase 1 right after parse_trajectory; Phase 1.5 and Phase 2
# read the resulting pool from the Phase-1 output JSON via the new
# ``step_compressions`` field and never re-compress.
# =====================================================================


def _format_step_line(
    step_num: Any,
    agent_text: str,
    env_text: str,
    role: Optional[str] = None,
) -> str:
    """Render a single precomputed step into the canonical line shape.

    When ``role`` is non-empty the header becomes ``Step N [role: X]:`` so
    downstream prompts (Phase 1 history, Phase 2 trajectory block) can
    tell which agent / speaker produced each step. ``role`` is typically
    ``message_role`` (the fine-grained ``name`` when present, e.g.
    ``Orchestrator`` / ``WebSurfer`` for whoandwhen; ``user`` / ``assistant``
    for alfworld / gaia / webshop).

    Shapes:
    - Both sides present  -> "Step N [role: X]:\\n  Agent: ...\\n  Env: ..."
    - Only one side       -> "Step N [role: X]:\\n  Agent: ..."  or  "Step N [role: X]:\\n  Env: ..."
    - Both empty          -> "Step N [role: X]: (empty)"
    - ``role`` missing    -> header drops the ``[role: ...]`` bracket.
    """
    agent_text = (agent_text or "").strip()
    env_text = (env_text or "").strip()
    role_str = (role or "").strip()
    header = f"Step {step_num}"
    if role_str:
        header += f" [role: {role_str}]"
    header += ":"
    parts: List[str] = [header]
    if agent_text:
        # parts.append(f"  Agent: {agent_text}")
        parts.append(agent_text)
    if env_text:
        # parts.append(f"  Env: {env_text}")
        parts.append(env_text)
    if len(parts) == 1:
        parts.append("  (empty)")
    return "\n".join(parts)


def _split_step_by_role(role: str, content: str) -> Tuple[str, str]:
    """Map a raw flat step (role, content) into (agent_text, env_text)
    using ``classify_message_kind``. Used for short-circuit and fallback
    paths so they emit the same canonical Agent/Env shape.
    """
    text = str(content or "")
    if not text.strip():
        return "", ""
    kind = classify_message_kind(role, text)
    if kind == "agent_cognition":
        return text, ""
    if kind == "env_feedback":
        return "", text
    return text, ""  # mixed / unknown -> bias to agent side, env left empty


def _build_chunk_compress_prompt(
    chunk: List[Dict[str, Any]],
    max_chars: int,
) -> str:
    lines = []
    for entry in chunk:
        lines.append(
            f"[step={entry.get('step')} role={entry.get('role','')}]\n{entry.get('content','')}"
        )
    payload = "\n\n".join(lines).strip() or "(empty)"
    return f"""
You are compressing {len(chunk)} consecutive trajectory step(s) for downstream
agent error diagnosis.

For EACH step, compress ONLY the information that actually occurred in that step.

Important distinction:
- agent-side content means content PRODUCED BY THE AGENT in this step:
  actual thoughts, plans, reflections, decisions, selected actions, claims, summaries,
  or tool calls that the agent already made.
- env-side content means feedback PRODUCED BY THE ENVIRONMENT or external tools:
  observations, tool outputs, page dumps, terminal output, error messages, URLs,
  ids, numbers, truncation markers, admissible actions if they are part of the
  environment state.

Do NOT treat user instructions as agent content.
Do NOT put requested future sections such as "<memory>", "<reflection>", "<plan>",
or "<action>" into agent unless the agent has actually generated them.
Do NOT preserve formatting instructions, output schemas, or role/task boilerplate
unless they are themselves the object being diagnosed.
Do NOT invent missing agent cognition.

When a user prompt contains a trajectory wrapper, such as:
- past observations/actions,
- current observation,
- admissible actions,
- instructions telling the agent how to respond,

then classify only the actual trajectory information:
- current observation -> env
- previous observations/actions -> env only if needed for diagnosing this step
- admissible actions -> env only if needed
- user instructions / output format / requested reasoning structure -> omit
- agent field -> omit unless the step contains actual agent-generated content

Prefer the minimal sufficient compression for error diagnosis.
If the current step has no actual agent-produced content, omit the agent field.

================ PRESERVE VERBATIM =============
Downstream error diagnosis cites specific substrings of this compressed text
word-for-word as evidence. If the original phrase is edited during compression,
that evidence check fails. Therefore preserve the FOLLOWING classes of tokens
VERBATIM whenever they appear in the input (copy them into the agent/env field
exactly as written, do not translate, paraphrase, reorder, round, or abbreviate):

- Proper nouns and named entities (people, places, brands, franchise/universe
  names such as "Disney", "Marvel", "MCU", book/paper/file names).
- Product / part / SKU / ASIN / ISBN / DOI / arXiv identifiers and any
  alphanumeric IDs the environment returned.
- Numbers with their units exactly as given (prices like "$22.50", quantities
  like "3 items", measurements like "45 min", percentages, counts).
- Years and dates (including citation years like "(1976)", ISO dates, "Q3
  2023"-style period labels).
- URLs, file paths, CLI flags, API parameter names and their literal values.
- Quoted constraint phrases from the task statement (e.g. "avoid Tue",
  "refundable", "under $25", "first page only") — keep them inside quotation
  marks if the original had them.
- Tool / function names and their exact argument keys.
- Error messages, status codes, truncation markers ("has_more: true",
  "output truncated", "403 Forbidden").

You MAY rewrite the connective prose between these tokens to shorten the field;
you MAY NOT rewrite the tokens themselves. If preserving them verbatim would
push a field past the length cap, DROP the least-important surrounding prose
first and keep the verbatim tokens.

================ STEPS =============
{payload}

Return ONLY one JSON object, no commentary, no code fences:
{{
  "steps": [
    {{"step": <int>, "agent": "<<= {max_chars} chars or empty string>", "env": "<<= {max_chars} chars or empty string>"}}
  ]
}}

================ OUTPUT CONSTRAINTS =============

Constraints:
- Each step object MUST contain exactly these three fields: "step", "agent", and "env".
- If the current step has no agent-side content, set "agent": "".
- If the current step has no env-side content, set "env": "".
- Do NOT omit "agent" or "env".
- Each non-empty agent/env field MUST be <= {max_chars} characters.
- The "steps" array MUST contain exactly one entry per input step in order.
"""


def _parse_chunk_compress_response(
    response_text: str,
    chunk: List[Dict[str, Any]],
    max_chars: int,
) -> Optional[Dict[int, Tuple[str, str]]]:
    """Parse the LLM response into ``{step_num: (agent_text, env_text)}``.

    Returns ``None`` on any parse failure so the caller can fall back.
    """
    payload = strip_think_tags(str(response_text or "")).strip()
    if not payload:
        return None
    payload = re.sub(r"^```(?:json)?\s*|\s*```$", "", payload, flags=re.IGNORECASE | re.DOTALL).strip()
    obj_text = _extract_json_object(payload) or payload
    try:
        obj = json.loads(obj_text)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    items = obj.get("steps")
    if not isinstance(items, list):
        return None

    out: Dict[int, Tuple[str, str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        sid_raw = item.get("step")
        try:
            sid = int(sid_raw)
        except Exception:
            continue
        agent = str(item.get("agent", "") or "").strip()
        env = str(item.get("env", "") or "").strip()
        # if agent and len(agent) > max_chars:
        #     agent = clip_text_middle(agent, max_chars)
        if env and len(env) > max_chars:
            env = clip_text_middle(env, max_chars)
        out[sid] = (agent, env)

    for entry in chunk:
        try:
            sid = int(entry.get("step"))
        except Exception:
            continue
        if sid not in out:
            return None
    return out


async def _compress_chunk_async(
    agenerate_fn: Callable[..., Awaitable[Any]],
    chunk: List[Dict[str, Any]],
    max_chars: int,
    max_tokens: Optional[int] = None,
    force_json: bool = True,
    max_lm_tokens: int = DEFAULT_COMPRESS_MAX_LM_TOKENS,
) -> Dict[int, Tuple[str, str]]:
    """Compress one chunk of steps. Short-circuits when every step's raw
    content is already within ``max_chars`` (skips LLM entirely).

    ``max_tokens`` auto-scales from ``max_chars * len(chunk)`` via
    :func:`estimate_compress_max_tokens` when not supplied. This
    replaces the previous hardcoded 1024 which silently truncated th1
    / th2 responses on larger chunks (the parse would then fail and
    the tier would fall back to ``clip_text_middle`` without warning).

    ``force_json`` controls whether the chunk-compress call pins
    ``response_format={"type": "json_object"}``. Set to ``False`` for
    reasoning-model backends whose quality drops under strict JSON
    mode; the parser then runs :func:`strip_think_tags` before JSON
    extraction and still falls back cleanly on parse failure.

    ``max_lm_tokens`` is the backend context-length cap (input +
    output) in tokens. When the estimated ``input_tokens + max_tokens``
    would exceed this cap, the chunk is **halved and recursively
    compressed** in parallel; if the chunk already contains a single
    step and still overflows, that step's ``content`` is first
    :func:`clip_text_middle`-truncated to the remaining budget before
    the LLM call. Both events are logged at WARNING.

    On LLM / parse failure, falls back per-step to ``clip_text_middle``
    on the raw text routed by ``classify_message_kind``.
    """
    short_circuit_all = True
    for entry in chunk:
        if len(str(entry.get("content", "") or "")) > max_chars:
            short_circuit_all = False
            break

    if short_circuit_all:
        result: Dict[int, Tuple[str, str]] = {}
        for entry in chunk:
            sid = int(entry.get("step", 0))
            agent_text, env_text = _split_step_by_role(
                str(entry.get("role", "") or ""), str(entry.get("content", "") or "")
            )
            result[sid] = (agent_text, env_text)
        return result

    if max_tokens is None:
        max_tokens = estimate_compress_max_tokens(max_chars, len(chunk))

    # --- Context-length (input + output) budget enforcement ---------
    # If the rendered prompt + expected output would blow past the
    # backend's ``max_lm_tokens``, first try to halve the chunk (keeps
    # batched-compression benefit); fall back to clipping the single
    # remaining step's raw content when len(chunk) == 1.
    est_input_tokens = _estimate_chunk_input_tokens(chunk)
    if est_input_tokens + int(max_tokens) > int(max_lm_tokens):
        if len(chunk) > 1:
            mid = len(chunk) // 2
            sid_lo = chunk[0].get("step")
            sid_hi = chunk[-1].get("step")
            logger.warning(
                "compress chunk over context window: steps=[%s..%s] size=%d "
                "est_input_tokens=%d + max_tokens=%d > max_lm_tokens=%d; "
                "splitting in half",
                sid_lo, sid_hi, len(chunk), est_input_tokens, int(max_tokens), int(max_lm_tokens),
            )
            left_task = _compress_chunk_async(
                agenerate_fn, chunk[:mid], max_chars,
                max_tokens=None, force_json=force_json,
                max_lm_tokens=max_lm_tokens,
            )
            right_task = _compress_chunk_async(
                agenerate_fn, chunk[mid:], max_chars,
                max_tokens=None, force_json=force_json,
                max_lm_tokens=max_lm_tokens,
            )
            left_res, right_res = await asyncio.gather(left_task, right_task)
            merged: Dict[int, Tuple[str, str]] = {}
            merged.update(left_res)
            merged.update(right_res)
            return merged
        else:
            # Single-step chunk still over budget: clip its content.
            entry = chunk[0]
            raw_content = str(entry.get("content", "") or "")
            budget_tokens = (
                int(max_lm_tokens)
                - int(max_tokens)
                - int(COMPRESS_PROMPT_TEMPLATE_OVERHEAD_CHARS / COMPRESS_MAX_TOKENS_CHARS_PER_TOKEN)
                - 64  # per-entry header + safety
            )
            # Convert surviving token budget back to chars; floor at
            # 2*max_chars so we never clip below what the tier aims to
            # produce on the output side.
            budget_chars = max(
                2 * int(max_chars),
                int(budget_tokens * COMPRESS_MAX_TOKENS_CHARS_PER_TOKEN),
            )
            if len(raw_content) > budget_chars:
                logger.warning(
                    "compress single-step chunk over context window: step=%s "
                    "est_input_tokens=%d + max_tokens=%d > max_lm_tokens=%d; "
                    "clipping content %d -> %d chars",
                    entry.get("step"), est_input_tokens, int(max_tokens),
                    int(max_lm_tokens), len(raw_content), budget_chars,
                )
                clipped_entry = dict(entry)
                clipped_entry["content"] = clip_text_middle(raw_content, budget_chars)
                chunk = [clipped_entry]

    prompt = _build_chunk_compress_prompt(chunk, max_chars)
    response = ""
    try:
        gen_kwargs: Dict[str, Any] = {"max_tokens": max_tokens, "temperature": 0.0}
        if force_json:
            gen_kwargs["response_format"] = {"type": "json_object"}
        response = await agenerate_fn(prompt, **gen_kwargs)
    except Exception:
        response = ""

    parsed = _parse_chunk_compress_response(str(response or ""), chunk, max_chars)
    if parsed is not None:
        return parsed

    fallback: Dict[int, Tuple[str, str]] = {}
    for entry in chunk:
        sid = int(entry.get("step", 0))
        agent_text, env_text = _split_step_by_role(
            str(entry.get("role", "") or ""), str(entry.get("content", "") or "")
        )
        if agent_text and len(agent_text) > max_chars:
            agent_text = clip_text_middle(agent_text, max_chars)
        if env_text and len(env_text) > max_chars:
            env_text = clip_text_middle(env_text, max_chars)
        fallback[sid] = (agent_text, env_text)
    return fallback


async def _compute_tier(
    agenerate_fn: Callable[..., Awaitable[Any]],
    flat_steps: List[Dict[str, Any]],
    max_chars: int,
    chunk_size: int,
    max_tokens: Optional[int] = None,
    force_json: bool = True,
    max_lm_tokens: int = DEFAULT_COMPRESS_MAX_LM_TOKENS,
) -> Dict[int, str]:
    """Run one tier (th1 / th2 / th3) over the whole trajectory.

    ``flat_steps`` items are normalized dicts with keys
    ``{"step", "role", "content"}``.

    ``max_tokens`` is passed through to each chunk's compression call;
    when ``None`` (default), ``_compress_chunk_async`` auto-estimates
    it from ``max_chars * len(chunk)``.

    ``force_json`` controls strict JSON response format at every chunk
    call; see :func:`_compress_chunk_async` for the free-form-then-parse
    path used when set to ``False``.

    ``max_lm_tokens`` is the backend context-length cap (input + output)
    in tokens; passed through so each chunk can adaptively split / clip.
    """
    if not flat_steps:
        return {}
    chunk_size = max(1, int(chunk_size))
    chunks: List[List[Dict[str, Any]]] = [
        flat_steps[i:i + chunk_size]
        for i in range(0, len(flat_steps), chunk_size)
    ]
    tasks = [
        _compress_chunk_async(
            agenerate_fn,
            c,
            max_chars=max_chars,
            max_tokens=max_tokens,
            force_json=force_json,
            max_lm_tokens=max_lm_tokens,
        )
        for c in chunks
    ]
    results = await asyncio.gather(*tasks)

    role_by_sid: Dict[int, str] = {}
    for entry in flat_steps:
        try:
            sid = int(entry.get("step"))
        except Exception:
            continue
        role_by_sid[sid] = str(entry.get("role", "") or "")

    rendered: Dict[int, str] = {}
    for chunk_result in results:
        for sid, (agent_text, env_text) in chunk_result.items():
            rendered[sid] = _format_step_line(
                sid, agent_text, env_text, role=role_by_sid.get(sid)
            )
    return rendered


async def aprecompute_step_compressions(
    agenerate_fn: Callable[..., Awaitable[Any]],
    flat_steps: List[Dict[str, Any]],
    th1_max: int = DEFAULT_STEP_TH1_MAX_CHARS,
    th2_max: int = DEFAULT_STEP_TH2_MAX_CHARS,
    th3_max: int = DEFAULT_STEP_TH3_MAX_CHARS,
    th1_chunk: int = DEFAULT_TH1_CHUNK_SIZE,
    th2_chunk: int = DEFAULT_TH2_CHUNK_SIZE,
    th3_chunk: int = DEFAULT_TH3_CHUNK_SIZE,
    th1_max_tokens: Optional[int] = None,
    th2_max_tokens: Optional[int] = None,
    th3_max_tokens: Optional[int] = None,
    force_json: bool = True,
    max_lm_tokens: int = DEFAULT_COMPRESS_MAX_LM_TOKENS,
) -> Dict[int, Dict[str, str]]:
    """Precompute per-step Agent/Env-split compressions at three tiers.

    ``flat_steps`` must be a list of dicts with at least
    ``{"step": int, "message_role": str, "content": str}`` (role may
    also be supplied via the ``role`` key).

    Returns a dict ``{step_num: {"th1": "...", "th2": "...", "th3": "..."}}``
    where each value is a single string already shaped as

        Step N:
          Agent: ...
          Env: ...

    (omitting the side that has no content). Steps whose raw content is
    already within a tier's per-step cap short-circuit to verbatim role-
    routed text without any LLM call. If every step in a chunk short-
    circuits, that chunk skips the LLM entirely.

    Per-tier ``max_tokens`` default to ``None`` so the compression call
    auto-scales with ``max_chars * chunk_size`` (see
    :func:`estimate_compress_max_tokens`). Supply explicit integers
    only when a specific backend needs a different ceiling than the
    auto-estimate; otherwise leaving these unset is the recommended
    path and avoids silent truncation at finish_reason=length.
    """
    if not flat_steps:
        return {}

    normalized: List[Dict[str, Any]] = []
    for entry in flat_steps:
        try:
            sid = int(entry.get("step", 0))
        except Exception:
            continue
        normalized.append({
            "step": sid,
            "role": str(entry.get("message_role", "") or entry.get("role", "") or ""),
            "content": str(entry.get("content", "") or ""),
        })
    if not normalized:
        return {}

    th1_task = _compute_tier(
        agenerate_fn, normalized, th1_max, th1_chunk,
        max_tokens=th1_max_tokens, force_json=force_json,
        max_lm_tokens=max_lm_tokens,
    )
    th2_task = _compute_tier(
        agenerate_fn, normalized, th2_max, th2_chunk,
        max_tokens=th2_max_tokens, force_json=force_json,
        max_lm_tokens=max_lm_tokens,
    )
    th3_task = _compute_tier(
        agenerate_fn, normalized, th3_max, th3_chunk,
        max_tokens=th3_max_tokens, force_json=force_json,
        max_lm_tokens=max_lm_tokens,
    )
    th1_map, th2_map, th3_map = await asyncio.gather(th1_task, th2_task, th3_task)

    pool: Dict[int, Dict[str, str]] = {}
    for entry in normalized:
        sid = entry["step"]
        fallback_line = _format_step_line(
            sid,
            *_split_step_by_role(entry["role"], entry["content"]),
            role=entry["role"],
        )
        pool[sid] = {
            "th1": th1_map.get(sid, fallback_line),
            "th2": th2_map.get(sid, fallback_line),
            "th3": th3_map.get(sid, fallback_line),
        }
    return pool


# =====================================================================
# Unified per-step three-tier compression (single LLM call per step)
# =====================================================================
#
# Instead of running three independent tier computations (one LLM call
# per chunk per tier), the unified path asks the LLM to produce th1 /
# th2 / th3 compressed views in ONE call per step, reducing total calls
# from  N + ceil(N/3) + ceil(N/5)  to  N.
#
# The old per-tier functions (_compute_tier, _compress_chunk_async, etc.)
# are kept intact for backward-compat via ``unified=False``.
# =====================================================================


def _build_unified_compress_prompt(
    entry: Dict[str, Any],
    th1_max: int,
    th2_max: int,
    th3_max: int,
) -> str:
    """Build a prompt that asks the LLM to produce all three tiers at once.

    ``entry`` is a normalized dict ``{"step": int, "role": str, "content": str}``.
    Returns the full prompt string.
    """
    step_num = entry.get("step", 0)
    role = entry.get("role", "")
    content = entry.get("content", "") or "(empty)"
    payload = f"[step={step_num} role={role}]\n{content}"

    return f"""You are compressing a single trajectory step at THREE different compression levels for downstream agent error diagnosis. Produce a detailed version (th1), a moderate version (th2), and a concise version (th3) of the same step content.

================ ROLE CLASSIFICATION =============

For this step, split content into two fields:
- "agent": content PRODUCED BY THE AGENT — actual thoughts, plans, reflections, decisions, selected actions, claims, summaries, or tool calls the agent already made.
- "env": feedback PRODUCED BY THE ENVIRONMENT or external tools — observations, tool outputs, page dumps, terminal output, error messages, URLs, IDs, numbers, truncation markers, admissible actions that are part of the environment state.

Do NOT treat user instructions as agent content.
Do NOT put requested future sections such as "<memory>", "<reflection>", "<plan>", or "<action>" into agent unless the agent has actually generated them.
Do NOT preserve formatting instructions, output schemas, or role/task boilerplate unless they are themselves the object being diagnosed.
Do NOT invent missing agent cognition.

When a user prompt contains a trajectory wrapper, such as:
- past observations/actions,
- current observation,
- admissible actions,
- instructions telling the agent how to respond,

then classify only the actual trajectory information:
- current observation -> env
- previous observations/actions -> env only if needed for diagnosing this step
- admissible actions -> env only if needed
- user instructions / output format / requested reasoning structure -> omit
- agent field -> omit unless the step contains actual agent-generated content

================ CORE COMPRESSION PRINCIPLES =============

1. PRESERVE SEMANTIC CORE: Compression removes ONLY redundant phrasing, verbose boilerplate, repetitive expressions, and filler words. The core meaning of EVERY sentence must survive. When in doubt, keep more rather than less.

2. CONSTRAINTS AND PLANS ARE CRITICAL: If the content contains ANY of the following, their key information MUST be preserved across ALL three tiers:
   - Constraints (time limits, budget limits, "do NOT do X" prohibitions, "must do Y first" preconditions, required formats, allowed/disallowed actions)
   - Plans (specific action items, ordered steps, goals, sub-goals, milestones)
   - Requirements for subsequent steps (instructions that govern future behavior, expected outputs, acceptance criteria)
   - Conditional logic ("if X then Y", fallback strategies, error handling rules)
   Drop surrounding prose before dropping any constraint or plan item.

3. THREE-TIER COMPRESSION LEVELS:
   - th1 (detailed, <={th1_max} chars per field): Preserve all meaningful details. Remove only obvious redundancy, repeated phrasings, and decorative formatting. Keep every distinct fact, constraint, action, and observation.
   - th2 (moderate, <={th2_max} chars per field): Merge related information. Omit secondary details and elaborations. But KEEP all constraints, key decisions, action outcomes, error messages, and critical observations.
   - th3 (concise, <={th3_max} chars per field): Retain only the most essential actions, results, errors, and constraints. Use minimal wording. Still preserve all hard constraints and critical factual anchors.

================ PRESERVE VERBATIM =============

Downstream error diagnosis cites specific substrings of this compressed text word-for-word as evidence. If the original phrase is edited during compression, that evidence check fails. Therefore preserve the FOLLOWING classes of tokens VERBATIM (copy them exactly as written — do not translate, paraphrase, reorder, round, or abbreviate):

- Proper nouns and named entities (people, places, brands, franchise/universe names such as "Disney", "Marvel", "MCU", book/paper/file names).
- Product / part / SKU / ASIN / ISBN / DOI / arXiv identifiers and any alphanumeric IDs the environment returned.
- Numbers with their units exactly as given (prices like "$22.50", quantities like "3 items", measurements like "45 min", percentages, counts).
- Years and dates (including citation years like "(1976)", ISO dates, "Q3 2023"-style period labels).
- URLs, file paths, CLI flags, API parameter names and their literal values.
- Quoted constraint phrases from the task statement (e.g. "avoid Tue", "refundable", "under $25", "first page only") — keep them inside quotation marks if the original had them.
- Tool / function names and their exact argument keys.
- Error messages, status codes, truncation markers ("has_more: true", "output truncated", "403 Forbidden").

You MAY rewrite the connective prose between these tokens to shorten the field; you MAY NOT rewrite the tokens themselves. If preserving them verbatim would push a field past the length cap, DROP the least-important surrounding prose first and keep the verbatim tokens.

================ STEP =============
{payload}

================ OUTPUT =============

Return ONLY one JSON object, no commentary, no code fences:
{{
  "th1": {{"agent": "<=  {th1_max} chars or empty string", "env": "<= {th1_max} chars or empty string"}},
  "th2": {{"agent": "<= {th2_max} chars or empty string", "env": "<= {th2_max} chars or empty string"}},
  "th3": {{"agent": "<= {th3_max} chars or empty string", "env": "<= {th3_max} chars or empty string"}}
}}

================ OUTPUT CONSTRAINTS =============

- The JSON object MUST have exactly three keys: "th1", "th2", "th3".
- Each tier object MUST have exactly two string fields: "agent" and "env".
- If this step has no agent-side content, set "agent": "".
- If this step has no env-side content, set "env": "".
- th1 agent/env fields MUST each be <= {th1_max} characters.
- th2 agent/env fields MUST each be <= {th2_max} characters.
- th3 agent/env fields MUST each be <= {th3_max} characters.
- th1 should be the most detailed, th3 the most concise. th2 is in between.
"""


def _parse_unified_compress_response(
    response_text: str,
    entry: Dict[str, Any],
    th1_max: int,
    th2_max: int,
    th3_max: int,
) -> Optional[Dict[str, Tuple[str, str]]]:
    """Parse the unified LLM response into ``{"th1": (agent, env), ...}``.

    Returns ``None`` on any parse failure so the caller can fall back.
    """
    payload = strip_think_tags(str(response_text or "")).strip()
    if not payload:
        return None
    payload = re.sub(
        r"^```(?:json)?\s*|\s*```$", "", payload,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    obj_text = _extract_json_object(payload) or payload
    try:
        obj = json.loads(obj_text)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None

    tier_caps = {"th1": th1_max, "th2": th2_max, "th3": th3_max}
    out: Dict[str, Tuple[str, str]] = {}
    for tier, cap in tier_caps.items():
        tier_obj = obj.get(tier)
        if not isinstance(tier_obj, dict):
            return None
        agent = str(tier_obj.get("agent", "") or "").strip()
        env = str(tier_obj.get("env", "") or "").strip()
        if agent and len(agent) > cap:
            agent = clip_text_middle(agent, cap)
        if env and len(env) > cap:
            env = clip_text_middle(env, cap)
        out[tier] = (agent, env)

    return out


def _is_code_or_terminal(content: str) -> bool:
    """Heuristic: detect if content is code, terminal output, or other
    machine-generated text that doesn't benefit from LLM compression.

    Returns ``True`` when the content is predominantly structured /
    machine-generated, meaning deterministic ``clip_text_middle`` is
    sufficient (and much cheaper than an LLM call).
    """
    if not content or not content.strip():
        return False

    text = content.strip()

    # --- Strong signals (any one is enough) ---
    # Diff / patch output
    if "diff --git" in text or text.startswith("---") and "+++ " in text[:500]:
        return True
    # Traceback
    if "Traceback (most recent call last)" in text:
        return True

    # --- Line-level heuristic ---
    lines = text.split("\n")
    if len(lines) < 3:
        return False

    code_indicators = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Indented lines (code / structured output)
        if line.startswith(("    ", "\t")):
            code_indicators += 1
        # Common code patterns
        elif stripped.startswith(("def ", "class ", "import ", "from ", "return ",
                                  "if ", "elif ", "else:", "for ", "while ",
                                  "try:", "except ", "raise ", "with ",
                                  "async ", "await ", "#", "//", "/*", "*/",
                                  "function ", "const ", "let ", "var ",
                                  "export ", "module.")):
            code_indicators += 1
        # Terminal / shell output patterns
        elif stripped.startswith(("$ ", "> ", ">>> ", "... ", "+ ", "- ",
                                  "drwx", "-rw-", "total ", "commit ",
                                  "Author:", "Date:", "@@")):
            code_indicators += 1
        # File paths as lines
        elif stripped.startswith(("./", "/")) and " " not in stripped[:60]:
            code_indicators += 1
        # grep -n / cat -n output: "123:code" or "path/file.py:123:code"
        # Also matches "  123  code" (cat -n with leading spaces)
        elif re.match(r"^\d+[:\-]", stripped):
            code_indicators += 1
        elif re.match(r"^\d+\s{2,}\S", stripped):
            code_indicators += 1
        elif re.match(r"^[\w./_-]+\.(?:py|js|ts|java|c|cpp|h|rs|go|rb|sh|yaml|yml|json|xml|html|css|sql|toml|cfg|ini|txt|md):\d+", stripped):
            code_indicators += 1
        # Status markers from tool wrappers
        elif any(marker in stripped for marker in (
            "[The command completed",
            "exit code",
            "PASSED", "FAILED", "ERROR",
            "===", "---",
        )):
            code_indicators += 1

    non_empty = sum(1 for l in lines if l.strip())
    if non_empty == 0:
        return False

    return code_indicators / non_empty > 0.4


async def _compress_step_unified_async(
    agenerate_fn: Callable[..., Awaitable[Any]],
    entry: Dict[str, Any],
    th1_max: int,
    th2_max: int,
    th3_max: int,
    stage_a_max_tokens: int = DEFAULT_STAGE_A_MAX_TOKENS,
    force_json: bool = True,
    max_lm_tokens: int = DEFAULT_COMPRESS_MAX_LM_TOKENS,
) -> Dict[str, Tuple[str, str]]:
    """Compress one step into three tiers in a single LLM call.

    Short-circuits when raw content is within ALL tier caps.
    Falls back to ``_split_step_by_role`` + ``clip_text_middle`` on failure.

    Returns ``{"th1": (agent, env), "th2": (agent, env), "th3": (agent, env)}``.
    """
    content = str(entry.get("content", "") or "")
    role = str(entry.get("role", "") or "")
    sid = int(entry.get("step", 0))

    # -- Short-circuit: content already within the LARGEST tier cap -----
    # If content <= th1_max, all tiers can use the same raw split.
    # If content <= some tier but not all, we still call LLM (it will
    # produce all three tiers; the LLM may compress better than clip).
    if len(content) <= th3_max:
        agent_text, env_text = _split_step_by_role(role, content)
        return {
            "th1": (agent_text, env_text),
            "th2": (agent_text, env_text),
            "th3": (agent_text, env_text),
        }

    # -- Short-circuit: compress_role="compress" with code/terminal -----
    # Steps marked compress_role="compress" are typically tool outputs
    # (terminal output, code, file listings, diffs, test results).
    # If the content looks like machine-generated output, skip the LLM
    # and use deterministic clip_text_middle instead — it's faster and
    # the LLM adds little value for structured / code content.
    compress_role = str(entry.get("compress_role", "") or "")
    if compress_role == "compress" and _is_code_or_terminal(content):
        agent_text, env_text = _split_step_by_role(role, content)
        fallback: Dict[str, Tuple[str, str]] = {}
        for tier, cap in [("th1", th1_max), ("th2", th2_max), ("th3", th3_max)]:
            a = clip_text_middle(agent_text, cap) if agent_text and len(agent_text) > cap else agent_text
            e = clip_text_middle(env_text, cap) if env_text and len(env_text) > cap else env_text
            fallback[tier] = (a, e)
        return fallback

    # -- Context-window budget enforcement -----------------------------
    work_entry = dict(entry)
    est_input_tokens = _estimate_chunk_input_tokens([work_entry])
    if est_input_tokens + int(stage_a_max_tokens) > int(max_lm_tokens):
        budget_tokens = (
            int(max_lm_tokens)
            - int(stage_a_max_tokens)
            - int(COMPRESS_PROMPT_TEMPLATE_OVERHEAD_CHARS / COMPRESS_MAX_TOKENS_CHARS_PER_TOKEN)
            - 64
        )
        budget_chars = max(
            2 * int(th1_max),
            int(budget_tokens * COMPRESS_MAX_TOKENS_CHARS_PER_TOKEN),
        )
        if len(content) > budget_chars:
            logger.warning(
                "unified compress step=%s over context window: "
                "est_input=%d + max_tokens=%d > max_lm_tokens=%d; "
                "clipping content %d -> %d chars",
                sid, est_input_tokens, stage_a_max_tokens,
                max_lm_tokens, len(content), budget_chars,
            )
            work_entry = dict(entry)
            work_entry["content"] = clip_text_middle(content, budget_chars)

    # -- Build prompt and call LLM -------------------------------------
    prompt = _build_unified_compress_prompt(work_entry, th1_max, th2_max, th3_max)
    response = ""
    try:
        gen_kwargs: Dict[str, Any] = {
            "max_tokens": stage_a_max_tokens,
            "temperature": 0.0,
        }
        if force_json:
            gen_kwargs["response_format"] = {"type": "json_object"}
        response = await agenerate_fn(prompt, **gen_kwargs)
    except Exception:
        logger.warning("unified compress step=%s LLM call failed", sid, exc_info=True)
        response = ""

    # -- Parse response ------------------------------------------------
    parsed = _parse_unified_compress_response(
        str(response or ""), entry, th1_max, th2_max, th3_max,
    )
    if parsed is not None:
        return parsed

    # -- Fallback: deterministic split + clip --------------------------
    logger.warning("unified compress step=%s parse failed, using fallback", sid)
    agent_text, env_text = _split_step_by_role(role, content)
    fallback: Dict[str, Tuple[str, str]] = {}
    for tier, cap in [("th1", th1_max), ("th2", th2_max), ("th3", th3_max)]:
        a = clip_text_middle(agent_text, cap) if agent_text and len(agent_text) > cap else agent_text
        e = clip_text_middle(env_text, cap) if env_text and len(env_text) > cap else env_text
        fallback[tier] = (a, e)
    return fallback


async def aprecompute_step_compressions_unified(
    agenerate_fn: Callable[..., Awaitable[Any]],
    flat_steps: List[Dict[str, Any]],
    th1_max: int = DEFAULT_STEP_TH1_MAX_CHARS,
    th2_max: int = DEFAULT_STEP_TH2_MAX_CHARS,
    th3_max: int = DEFAULT_STEP_TH3_MAX_CHARS,
    stage_a_max_tokens: int = DEFAULT_STAGE_A_MAX_TOKENS,
    force_json: bool = True,
    max_lm_tokens: int = DEFAULT_COMPRESS_MAX_LM_TOKENS,
) -> Dict[int, Dict[str, str]]:
    """Unified single-call-per-step three-tier compression.

    Same return type as :func:`aprecompute_step_compressions`::

        {step_num: {"th1": "Step N: ...", "th2": "...", "th3": "..."}}

    but each step issues at most ONE LLM call (producing all three tiers),
    instead of up to three separate calls across independent tier passes.

    ``stage_a_max_tokens`` is the LLM ``max_tokens`` cap passed to every
    compression call. Set high for reasoning models whose thinking tokens
    far exceed the compressed output size.
    """
    if not flat_steps:
        return {}

    # -- Normalize steps -----------------------------------------------
    normalized: List[Dict[str, Any]] = []
    for entry in flat_steps:
        try:
            sid = int(entry.get("step", 0))
        except Exception:
            continue
        normalized.append({
            "step": sid,
            "role": str(entry.get("message_role", "") or entry.get("role", "") or ""),
            "content": str(entry.get("content", "") or ""),
            "compress_role": str(entry.get("compress_role", "") or ""),
        })
    if not normalized:
        return {}

    # -- Launch one task per step in parallel ---------------------------
    tasks = [
        _compress_step_unified_async(
            agenerate_fn,
            entry,
            th1_max=th1_max,
            th2_max=th2_max,
            th3_max=th3_max,
            stage_a_max_tokens=stage_a_max_tokens,
            force_json=force_json,
            max_lm_tokens=max_lm_tokens,
        )
        for entry in normalized
    ]
    results = await asyncio.gather(*tasks)

    # -- Render each tier through _format_step_line --------------------
    pool: Dict[int, Dict[str, str]] = {}
    for entry, tier_dict in zip(normalized, results):
        sid = entry["step"]
        role = entry["role"]
        fallback_line = _format_step_line(
            sid,
            *_split_step_by_role(role, entry["content"]),
            role=role,
        )
        pool[sid] = {}
        for tier in ("th1", "th2", "th3"):
            if tier in tier_dict:
                agent_text, env_text = tier_dict[tier]
                pool[sid][tier] = _format_step_line(
                    sid, agent_text, env_text, role=role,
                )
            else:
                pool[sid][tier] = fallback_line
    return pool
