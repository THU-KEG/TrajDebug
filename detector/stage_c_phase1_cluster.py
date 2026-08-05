#!/usr/bin/env python3
"""Stage C — Phase 1 (v5): Trigger clustering into error instances.

Per v5 spec §5.4.1 / §5.6.2, Phase 1 receives the flat list of
per-step triggers from Stage B and groups them into ERROR INSTANCES
via a single LLM call. One instance = "the same underlying erroneous
content / behavior" reflected across one or more steps.

Inputs:
  * ``<stem>_stage_b.json`` containing ``step_triggers``.

Output: ``<output_dir>/<stem>_stage_c_phase1.json`` carrying:
  * ``instances`` — list of instance dicts (Phase 1 schema);
  * echoed ``step_triggers`` and ``step_compressions`` for Phase 2.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from _stage_common import (
    extract_last_json_object,
    load_step_compressions_from_payload,
    resolve_extra_params,
)
from utils.llm_compression import strip_think_tags
from utils.model import APIModel
from utils.trajectory_utils import (
    collect_trajectory_sources,
    output_stem_for_source,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_CATEGORIES = ("cat-1", "cat-2", "cat-3", "env")
ALLOWED_ATTRIBUTIONS = ("agent", "env")


# =====================================================================
# System prompt (v5 §5.6.2)
# =====================================================================

PHASE1_SYSTEM_PROMPT = r"""You are an error-instance clustering annotator. You receive a list of per-step error TRIGGERS detected on one trajectory. Group those triggers into ERROR INSTANCES, where one instance corresponds to "the same underlying erroneous content / behavior" repeated or reflected across one or more steps.

=== TERMINOLOGY ===

Each trigger contradicts a "conflict object" — the thing the wrong content is conflicting with. Following the spec we call this object I. Concretely, I can be one of:
  - a TASK clause (sub-task, deliverable, constraint);
  - a HISTORY object (a specific physical / virtual location, a specific tool output, a specific observation, a specific memory entry, a specific prior tool-call signature);
  - a SAME-MESSAGE premise (operands, listed items, restated premise);
  - a prior agent-self statement (plan, reflection, memory).

Two triggers are about the SAME I when they are conflicting with the same underlying thing — e.g. same desk in the same room, same URL, same file path, same tool/sub-goal pair, same observation being ignored, same TASK constraint being violated.

Two triggers are about DIFFERENT I's when:
  - they conflict with different observations / different TASK clauses / different self-claims;
  - they happen to share the taxonomy_tag string but the underlying object is different.

=== CLUSTERING RULES (spec §4.1.3) ===

(I-a) Same step, same I, multiple modules (e.g. the same TASK omission shows up in plan AND act AND verify within one message) -> ONE instance.

(I-b) Same step, DIFFERENT I (e.g. one TASK omission AND one arithmetic mistake in the same message) -> MULTIPLE instances.

(I-c) Across steps, same I, never repaired in between -> ONE instance with origin_step = first occurrence.

(I-d) Across steps, same I, but it was repaired and then re-fires
      later -> SPLIT into TWO instances right here: the first
      instance spans from the first fire to the repair step; the
      second instance starts from the re-fire step (its origin_step
      = re-fire step). If you cannot determine whether a repair
      happened in between, conservatively keep them in ONE instance
      (fall back to I-c).

(I-e) "Fabricated I" (an I the agent invented out of thin air) is identified by the verbatim claim that introduced it.

GOOD CLUSTERING SIGNALS:
  - repeatedly probing the same physical / virtual location;
  - repeatedly calling the same tool with the same target;
  - repeatedly ignoring the same observation / memory entry;
  - the same TASK constraint dropped across plan / act / verify modules within a window.

BAD CLUSTERING SIGNALS:
  - both triggers tagged obs.* but they ignore DIFFERENT observations;
  - both tagged act.WrongTool but the wrong tool / sub-goal differs;
  - same step containing one TASK omission AND one arithmetic mistake (different I);
  - both are search strategy errors, but step 7 searched too narrowly and step 13 searched a completely different target — different I, separate instances;
  - same type of error repeated on DIFFERENT sub-goals — different I, separate instances.

CLUSTERING PRINCIPLE: cluster by the SPECIFIC CONSTRAINT OBJECT I
  being violated, NOT by "same type of erroneous behavior". Two
  triggers sharing a taxonomy_tag but violating DIFFERENT constraint
  objects must be separate instances. Only merge when both triggers
  violate THE SAME specific TASK clause / CONTEXT object / agent
  self-claim.

=== PRECISION REQUIREMENTS FOR `what_is_being_violated` ===

Write `what_is_being_violated` as a CONCRETE pointer to the exact
object being contradicted. Vague descriptions like "the TASK clause",
"the user's intent", or "the task statement" alone are NOT acceptable.

  - For TASK-clause violations: quote or paraphrase the specific
    constraint / deliverable / format requirement the action
    contradicts.
  - For HISTORY-object violations: quote or paraphrase the specific
    observation / tool output / location / URL being contradicted,
    and reference the step where it first appeared.
  - For SAME-MESSAGE premise violations: name the operands / listed
    items being misused within the same message.
  - For prior-self violations: quote or paraphrase the prior
    plan / reflection / memory statement being contradicted.

=== ONE-SHOT PROBE GUIDANCE ===

If a trigger fires at exactly ONE step and the agent at later steps
neither references the same wrong belief nor takes another action
against the same I, keep it as a STANDALONE singleton instance. Do
NOT merge it with other unrelated triggers, and do NOT manufacture
a broader I to tie it into later steps. The instance should contain
just that single trigger with a concrete `what_is_being_violated`
pointing to the specific local violation.

=== HARD CONSTRAINT ===

Triggers with attribution="agent" and triggers with attribution="env" CANNOT be merged into the same instance.

Every input trigger index must end up in EXACTLY ONE instance's trigger_indices.

=== ENV INSTANCE MERGING ===

For triggers with attribution="env", apply the following same-source merging rule IN ADDITION to the general clustering rules above:

Two env triggers point at the SAME underlying environment anomaly (same I) when they share at least one of the following same-source signals:
  - Same registrable domain (e.g. both involve keyence.com.tw, regardless of subdomain);
  - Same tracking parameter key+value (utm_source=google, gclid=..., etc.);
  - Same HTTP error code (e.g. both hit a 429 or 503);
  - Same failure keyword: "rate limit", "timeout", "content filter", "empty payload" / "no results", "captcha", "access denied" / "forbidden", "not found";
  - Same taxonomy_tag (e.g. both are env.HijackOrOffTopicNavigation).

When two env triggers share any of the above signals, treat them as the SAME I and merge them into ONE instance, even if the surface action differs (e.g. one is a navigation step and the other is a tool call). The merged instance's origin_step = earliest trigger step, last_trigger_step = latest trigger step.

Do NOT merge env triggers that share only a generic/CDN domain (google.com, bing.com, youtube.com, wikipedia.org, github.com, etc.) — those are not same-source signals.

=== TRAJECTORY CONTEXT ===

You will also receive a TRAJECTORY STEP CONTENT block that renders every step of the trajectory. Steps that fired at least one trigger are shown at HIGH detail (th1 compression); steps that did not fire any trigger are shown at LOW detail (th3 compression) purely as background so you can trace "did a repair happen in between re-fires" (rule I-d).

Use the high-detail steps to verify whether two triggers point at the SAME underlying constraint object I (same URL / same desk / same observation / same TASK clause). Use the low-detail steps only to check whether a repair occurred between two candidate triggers of the same I; do NOT invent new triggers from them.

=== OUTPUT FORMAT ===

{
  "instances": [
    {
      "instance_id": <int, 0-indexed across this trajectory>,
      "trigger_indices": [<int>, ...],
      "origin_step": <int, smallest step among trigger_indices>,
      "last_trigger_step": <int, largest step among trigger_indices>,
      "attribution": "agent" | "env",
      "category": "cat-1" | "cat-2" | "cat-3" | "env",
      "error_content": "<one-sentence natural-language description of the same erroneous content/behavior>",
      "what_is_being_violated": "<one-sentence description of the I being violated>",
      "merged_reasoning": "<2-3 sentences explaining why these triggers share the same I>"
    }
  ]
}

If the input triggers array is empty, emit  {"instances": []}.
"""


# =====================================================================
# User prompt builder
# =====================================================================


def _build_user_prompt(
    agent_framework_description: str,
    trajectory_overview: str,
    stage_b_triggers: List[Dict[str, Any]],
    trajectory_block: str = "",
) -> str:
    parts = []
    if agent_framework_description.strip():
        parts.append(f"=== AGENT FRAMEWORK ===\n{agent_framework_description.strip()}")
    if trajectory_overview.strip():
        parts.append(f"=== TRAJECTORY OVERVIEW ===\n{trajectory_overview.strip()}")
    if trajectory_block.strip():
        parts.append(f"=== TRAJECTORY STEP CONTENT ===\n{trajectory_block.strip()}")
    triggers_json = json.dumps(stage_b_triggers, indent=2, ensure_ascii=False)
    parts.append(f"=== INPUT TRIGGERS ===\n{triggers_json}")
    return "\n\n".join(parts)

def _build_trajectory_overview(stage_b_payload: Dict[str, Any]) -> str:
    """Build a short overview of the trajectory for context."""
    lines = []
    lines.append(f"task_id: {stage_b_payload.get('task_id', '?')}")
    lines.append(f"environment: {stage_b_payload.get('environment', '?')}")
    lines.append(f"total_steps: {stage_b_payload.get('total_steps', '?')}")
    lines.append(f"task_success: {stage_b_payload.get('task_success', '?')}")
    lines.append(f"task_outcome: {stage_b_payload.get('task_outcome', '?')}")
    task_desc = str(stage_b_payload.get("task_description", ""))
    if len(task_desc) > 500:
        task_desc = task_desc[:500] + "..."
    lines.append(f"task_description: {task_desc}")
    return "\n".join(lines)


def _build_trajectory_block(
    triggers: List[Dict[str, Any]],
    step_pool: Dict[int, Dict[str, str]],
    total_steps: Optional[int],
) -> str:
    """Render every step of the trajectory:

      * steps that fired >=1 trigger -> th1 (high detail);
      * steps that did not fire any trigger -> th3 (low-detail background).

    Falls back tier-by-tier when a requested tier is missing in the pool
    (th1 -> th2 -> th3; th3 -> th2 -> th1). Steps without any pooled
    entry are skipped silently (Phase 1 does not reload raw trajectory).
    """
    # Collect the set of steps that actually fired a trigger.
    triggered_steps: set = set()
    for t in triggers:
        try:
            triggered_steps.add(int(t.get("step")))
        except Exception:
            continue

    # Determine which step ids to render. Prefer pool keys (they cover
    # every step Stage A compressed); fall back to a 0..total_steps-1
    # range if the pool is empty.
    if step_pool:
        all_sids = sorted(step_pool.keys())
    elif isinstance(total_steps, int) and total_steps > 0:
        all_sids = list(range(total_steps))
    else:
        all_sids = sorted(triggered_steps)
    if not all_sids:
        return ""

    parts: List[str] = []
    for sid in all_sids:
        entry = step_pool.get(sid) if step_pool else None
        if sid in triggered_steps:
            # High-detail view for steps that fired a trigger.
            if entry:
                body = entry.get("th1") or entry.get("th2") or entry.get("th3") or ""
            else:
                body = ""
            tag = "triggered"
        else:
            # Low-detail background for non-triggered steps.
            if entry:
                body = entry.get("th3") or entry.get("th2") or entry.get("th1") or ""
            else:
                body = ""
            tag = "background"
        if not body:
            # Nothing usable for this step — skip silently.
            continue
        parts.append(f"--- step={sid} [{tag}] ---")
        parts.append(body.rstrip())
        parts.append("")
    return "\n".join(parts).rstrip()


# =====================================================================
# Env same-source post-merge (Phase 1 recall backstop)
# =====================================================================
#
# Phase 1 relies on the LLM to merge triggers by shared violated object I.
# For env instances, however, the LLM tends to split repeated hits of the
# *same* environment anomaly (same domain / same ad source / same HTTP
# error) into separate instances when the surface action differs, which
# then starves Phase 2's wasted_step_count and the span backstop.
#
# This pass runs *after* the LLM-parsed instances are validated. It only
# touches instances whose attribution == "env": two env instances get
# merged if they share at least one signature token extracted from the
# trigger quotes / taxonomy_tag. Signature tokens currently cover:
#   * registrable-ish domains (foo.bar.com -> bar.com);
#   * url tracking params (utm_*, gclid);
#   * HTTP status codes (4xx / 5xx);
#   * a small set of failure keywords (rate limit, timeout, content
#     filter, empty payload);
#   * the taxonomy_tag itself (env.HijackOrOffTopicNavigation, ...).
# Two env instances with overlapping signature tokens are merged into
# one; trigger_indices / origin_step / last_trigger_step are recomputed.


_ENV_DOMAIN_RE = re.compile(r"\b([a-z0-9][a-z0-9-]*\.(?:[a-z0-9-]+\.)+[a-z]{2,})\b", re.IGNORECASE)
_ENV_UTM_RE = re.compile(r"\b(utm_[a-z_]+)=([A-Za-z0-9_.\-]+)", re.IGNORECASE)
_ENV_GCLID_RE = re.compile(r"\b(gclid|fbclid|msclkid)=([A-Za-z0-9_.\-]+)", re.IGNORECASE)
_ENV_HTTP_CODE_RE = re.compile(r"\b(?:status(?:\s*code)?|http|error)[^0-9]{0,6}([45]\d{2})\b", re.IGNORECASE)
_ENV_BARE_CODE_RE = re.compile(r"\b([45]\d{2})\b")

_ENV_KEYWORD_SIGNATURES: Tuple[Tuple[str, str], ...] = (
    ("rate limit", "kw:rate_limit"),
    ("rate-limit", "kw:rate_limit"),
    ("rate_limited", "kw:rate_limit"),
    ("too many requests", "kw:rate_limit"),
    ("timeout", "kw:timeout"),
    ("timed out", "kw:timeout"),
    ("content filter", "kw:content_filter"),
    ("content_filter", "kw:content_filter"),
    ("responsible ai", "kw:content_filter"),
    ("empty payload", "kw:empty_payload"),
    ("empty response", "kw:empty_payload"),
    ("no results", "kw:empty_payload"),
    ("no result", "kw:empty_payload"),
    ("captcha", "kw:captcha"),
    ("access denied", "kw:access_denied"),
    ("forbidden", "kw:access_denied"),
    ("not found", "kw:not_found"),
    ("404", "kw:not_found"),
)

# Common hosting / CDN / generic domains that must NOT be used as a
# same-source signature on their own — they show up in unrelated pages
# and would over-merge.
_ENV_DOMAIN_BLOCKLIST: Set[str] = {
    "google.com",
    "www.google.com",
    "bing.com",
    "www.bing.com",
    "duckduckgo.com",
    "youtube.com",
    "m.youtube.com",
    "wikipedia.org",
    "en.wikipedia.org",
    "github.com",
    "raw.githubusercontent.com",
    "stackoverflow.com",
    "amazon.com",
    "www.amazon.com",
    "twitter.com",
    "x.com",
    "facebook.com",
    "linkedin.com",
    "example.com",
    "schema.org",
    "w3.org",
    "mozilla.org",
}


def _registrable_domain(host: str) -> str:
    """Collapse a full host to a (rough) registrable domain.

    Not a public-suffix-list implementation; we just keep the last two
    labels for generic TLDs and the last three for common 2-level TLDs
    (co.uk, com.tw, com.cn, co.jp, ...). Good enough to collapse
    `subdomain.keyence.com.tw` and `www.keyence.com.tw` into the same
    signature.
    """
    host = (host or "").strip().lower().strip(".")
    if not host:
        return ""
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    # Heuristic 2-level TLD handling.
    two_level_tlds = {
        "co.uk", "ac.uk", "gov.uk", "org.uk",
        "com.tw", "com.cn", "com.hk", "com.sg", "com.au", "com.br",
        "co.jp", "co.kr", "co.in", "co.nz",
        "org.cn", "net.cn", "gov.cn",
    }
    tail_two = ".".join(parts[-2:])
    tail_three = ".".join(parts[-3:]) if len(parts) >= 3 else tail_two
    if tail_two in two_level_tlds and len(parts) >= 3:
        return tail_three
    return tail_two


def _extract_env_signature_tokens(text: str) -> Set[str]:
    """Extract a set of same-source signature tokens from a trigger blob.

    Blob = wrong_content_quote + reference_quote + taxonomy_tag joined
    with spaces. Tokens returned are already namespaced ("dom:...",
    "utm:...", "gclid:...", "http:...", "kw:..."), so cross-family
    collisions cannot produce a false merge.
    """
    out: Set[str] = set()
    if not text:
        return out
    lower = text.lower()

    for m in _ENV_DOMAIN_RE.finditer(lower):
        reg = _registrable_domain(m.group(1))
        if not reg or reg in _ENV_DOMAIN_BLOCKLIST:
            continue
        out.add(f"dom:{reg}")

    for m in _ENV_UTM_RE.finditer(lower):
        key = m.group(1).lower()
        val = m.group(2).lower()
        out.add(f"utm:{key}={val}")

    for m in _ENV_GCLID_RE.finditer(lower):
        # Only the *presence* of gclid-like tracking is a signature; the
        # value itself is per-click random. Namespace it so gclid from
        # unrelated pages collides deliberately — which is exactly the
        # "same ad delivery pipeline" signal we want.
        out.add(f"track:{m.group(1).lower()}")

    for m in _ENV_HTTP_CODE_RE.finditer(lower):
        out.add(f"http:{m.group(1)}")
    # Also accept bare 4xx/5xx tokens when the text itself looks like a
    # server-error / tool-error snippet (conservative: require the word
    # "error" or "status" somewhere in the blob).
    if "error" in lower or "status" in lower:
        for m in _ENV_BARE_CODE_RE.finditer(lower):
            out.add(f"http:{m.group(1)}")

    for needle, token in _ENV_KEYWORD_SIGNATURES:
        if needle in lower:
            out.add(token)

    return out


def _merge_env_same_source_instances(
    instances: List["Phase1Instance"],
    triggers: List[Dict[str, Any]],
) -> List["Phase1Instance"]:
    """Merge env instances that share a same-source signature token.

    Only merges within ``attribution == "env"`` and within the same
    ``category`` bucket (which for env is always ``"env"`` but we stay
    defensive). Other instances pass through unchanged. The merge is
    conservative: two instances merge iff their signature sets have a
    non-empty intersection, so cat-1/2/3 and unrelated env instances
    (e.g. rate-limit at one step vs. ad hijack on a different domain)
    remain separate.
    """
    if not instances:
        return instances

    # Index trigger quotes so we can build per-instance signature sets.
    def _trigger_blob(idx: int) -> str:
        if not (0 <= idx < len(triggers)):
            return ""
        t = triggers[idx] or {}
        return " ".join(
            str(t.get(k, "") or "")
            for k in ("wrong_content_quote", "reference_quote", "taxonomy_tag")
        )

    # Precompute signature sets for every env instance; non-env stays None.
    sig_sets: List[Optional[Set[str]]] = []
    for inst in instances:
        if inst.attribution != "env":
            sig_sets.append(None)
            continue
        blob = " ".join(_trigger_blob(i) for i in inst.trigger_indices)
        sig_sets.append(_extract_env_signature_tokens(blob))

    # Union-find over env instances keyed by signature overlap.
    n = len(instances)
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        si = sig_sets[i]
        if not si:
            continue
        for j in range(i + 1, n):
            sj = sig_sets[j]
            if not sj:
                continue
            if instances[i].attribution != instances[j].attribution:
                continue
            if instances[i].category != instances[j].category:
                continue
            if si & sj:
                _union(i, j)

    # Collect groups; any singleton group is a no-op.
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        root = _find(i)
        groups.setdefault(root, []).append(i)

    # Rebuild the instance list; preserve original order by group root.
    merged: List[Phase1Instance] = []
    seen_roots: Set[int] = set()
    for i in range(n):
        root = _find(i)
        if root in seen_roots:
            continue
        seen_roots.add(root)
        members = groups[root]
        if len(members) == 1:
            merged.append(instances[members[0]])
            continue

        # Multi-member merge. Base on the earliest instance (smallest
        # origin_step, breaking ties by smallest index).
        members_sorted = sorted(
            members,
            key=lambda idx: (instances[idx].origin_step, idx),
        )
        base = instances[members_sorted[0]]
        all_idx: List[int] = []
        for m in members_sorted:
            all_idx.extend(instances[m].trigger_indices)
        # Dedup while preserving order, then sort ascending for the
        # downstream Phase 2 contract (origin_step = min, last = max).
        dedup_idx = sorted({int(x) for x in all_idx})
        steps = []
        for idx in dedup_idx:
            if 0 <= idx < len(triggers):
                s = triggers[idx].get("step")
                if isinstance(s, (int, float)):
                    steps.append(int(s))
        origin = min(steps) if steps else base.origin_step
        last_step = max(steps) if steps else base.last_trigger_step

        # Signature summary for auditability.
        shared_sig: Set[str] = set()
        for m in members_sorted:
            s = sig_sets[m]
            if s:
                shared_sig = shared_sig | s if not shared_sig else shared_sig & s
        shared_preview = ", ".join(sorted(shared_sig)[:6]) if shared_sig else ""
        extra_note = (
            f" [env-same-source auto-merge across {len(members_sorted)} LLM "
            f"instances; shared signature tokens: {shared_preview}]"
        )
        merged_reasoning = (base.merged_reasoning or "").rstrip()
        if extra_note.strip() not in merged_reasoning:
            merged_reasoning = (merged_reasoning + extra_note).strip()

        merged.append(Phase1Instance(
            instance_id=base.instance_id,  # re-indexed below
            trigger_indices=dedup_idx,
            origin_step=int(origin),
            last_trigger_step=int(last_step),
            attribution=base.attribution,
            category=base.category,
            error_content=base.error_content,
            what_is_being_violated=base.what_is_being_violated,
            merged_reasoning=merged_reasoning,
        ))

    return merged


# =====================================================================
# Parser
# =====================================================================


@dataclass
class Phase1Instance:
    instance_id: int
    trigger_indices: List[int]
    origin_step: int
    last_trigger_step: int
    attribution: str
    category: str
    error_content: str
    what_is_being_violated: str
    merged_reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "trigger_indices": self.trigger_indices,
            "origin_step": self.origin_step,
            "last_trigger_step": self.last_trigger_step,
            "attribution": self.attribution,
            "category": self.category,
            "error_content": self.error_content,
            "what_is_being_violated": self.what_is_being_violated,
            "merged_reasoning": self.merged_reasoning,
        }


def _parse_phase1_response(
    response: str, triggers: List[Dict[str, Any]]
) -> List[Phase1Instance]:
    """Parse and validate Phase 1 LLM output."""
    response = strip_think_tags(response or "")
    if response.strip().startswith("```"):
        response = "\n".join(
            line for line in response.splitlines()
            if not line.strip().startswith("```")
        )
    data = extract_last_json_object(response, must_have_key="instances")
    if data is None:
        data = extract_last_json_object(response)
    if not isinstance(data, dict):
        return []

    raw_instances = data.get("instances") or []
    if not isinstance(raw_instances, list):
        return []

    trigger_steps = {i: t.get("step") for i, t in enumerate(triggers)}
    trigger_attrs = {i: t.get("attribution", "agent") for i, t in enumerate(triggers)}
    total_triggers = len(triggers)

    instances: List[Phase1Instance] = []
    used_indices: set = set()

    for idx, item in enumerate(raw_instances):
        if not isinstance(item, dict):
            continue
        trig_idx = item.get("trigger_indices") or []
        if not isinstance(trig_idx, list):
            continue
        trig_idx = [int(x) for x in trig_idx if isinstance(x, (int, float)) and 0 <= int(x) < total_triggers]
        if not trig_idx:
            continue

        attribution = str(item.get("attribution", "")).strip()
        if attribution not in ALLOWED_ATTRIBUTIONS:
            first_attr = trigger_attrs.get(trig_idx[0], "agent")
            attribution = first_attr

        # Validate: all triggers in this instance must share attribution
        valid_idx = [i for i in trig_idx if trigger_attrs.get(i, "agent") == attribution]
        if not valid_idx:
            continue

        steps_in_instance = [trigger_steps.get(i, 0) for i in valid_idx]
        origin = min(steps_in_instance) if steps_in_instance else 0
        last_step = max(steps_in_instance) if steps_in_instance else 0

        category = str(item.get("category", "")).strip()
        if category not in ALLOWED_CATEGORIES:
            category = "env" if attribution == "env" else "cat-2"

        instances.append(Phase1Instance(
            instance_id=idx,
            trigger_indices=valid_idx,
            origin_step=int(origin),
            last_trigger_step=int(last_step),
            attribution=attribution,
            category=category,
            error_content=str(item.get("error_content", "") or "").strip(),
            what_is_being_violated=str(item.get("what_is_being_violated", "") or "").strip(),
            merged_reasoning=str(item.get("merged_reasoning", "") or "").strip(),
        ))
        used_indices.update(valid_idx)

    # Rescue: any trigger not assigned to an instance becomes a singleton
    for i in range(total_triggers):
        if i not in used_indices:
            attr = trigger_attrs.get(i, "agent")
            step = trigger_steps.get(i, 0)
            cat = triggers[i].get("category", "cat-2") if i < len(triggers) else "cat-2"
            instances.append(Phase1Instance(
                instance_id=len(instances),
                trigger_indices=[i],
                origin_step=int(step),
                last_trigger_step=int(step),
                attribution=attr,
                category=cat,
                error_content=f"[singleton rescue] trigger {i}",
                what_is_being_violated="",
                merged_reasoning="Not assigned by LLM; auto-singleton.",
            ))
            used_indices.add(i)

    # Re-index instance_id sequentially
    for new_id, inst in enumerate(instances):
        inst.instance_id = new_id

    return instances


# =====================================================================
# Clusterer class
# =====================================================================


class Phase1Clusterer:
    """Trigger-to-instance clustering (v5 Phase 1)."""

    def __init__(self, api_config: Dict[str, Any]):
        self.config = api_config
        cache_url = api_config.get("cache_url")
        if not cache_url:
            raise ValueError("Missing cache_url in api_config.")
        self.model = APIModel(
            cache_url,
            api_config["base_url"],
            api_config["model"],
            api_config.get("api_key", "EMPTY"),
            extra_params=resolve_extra_params(api_config.get("model_profile")),
        )
        self.judgement_force_json = bool(api_config.get("judgement_force_json", True))
        # Global LLM concurrency limiter shared across every cluster
        # call. Combined with the file-level semaphore in run_batch this
        # gives the router a steady stream of in-flight requests while
        # bounding the total fan-out.
        self.semaphore = asyncio.Semaphore(
            max(1, int(api_config.get("llm_concurrency", 10)))
        )
        # Wall-clock timeout for a single LLM call (seconds). 180s is a
        # ~3x safety margin over a typical reasoning-model response.
        self.request_timeout = float(api_config.get("request_timeout", 180.0))

    def close(self) -> None:
        self.model.close()

    async def _call_llm(self, system: str, user: str, usage_acc: Optional[Dict[str, int]] = None) -> str:
        """Call the LLM with 3 retries and a per-attempt wall-clock timeout.

        Retry plan (only knobs that the user requested are perturbed;
        the prompt itself is unchanged):
          * attempt 1: base temperature, base max_tokens.
          * attempt 2: temperature += 0.3, max_tokens *= 1.5.
          * attempt 3: temperature += 0.3 (relative to base), base
            max_tokens.
        Each attempt is wrapped in ``asyncio.wait_for(timeout)`` so a
        stuck request cannot stall the whole batch. After 3 failures we
        return an empty string and let the caller fall back to the
        singleton-rescue path in ``_parse_phase1_response``.
        """
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        response_format = {"type": "json_object"} if self.judgement_force_json else None
        base_max_tokens = int(self.config.get("max_tokens", 8192))
        base_temp = float(self.config.get("temperature", 0.0))
        attempts = [
            (base_temp, base_max_tokens),
            (min(1.0, base_temp + 0.3), int(base_max_tokens * 1.5)),
            (min(1.0, base_temp + 0.3), base_max_tokens),
        ]

        for idx, (temp, max_tokens) in enumerate(attempts):
            try:
                async with self.semaphore:
                    result, usage = await asyncio.wait_for(
                        asyncio.to_thread(
                            self.model.generate_chat,
                            messages,
                            max_tokens,
                            temp,
                            response_format,
                            return_usage=True,
                        ),
                        timeout=self.request_timeout,
                    )
                if usage and usage_acc is not None:
                    usage_acc["input_tokens"] += usage.get("input_tokens", 0)
                    usage_acc["reasoning_tokens"] += usage.get("reasoning_tokens", 0)
                    usage_acc["output_tokens"] += usage.get("output_tokens", 0)
                if result:
                    return result
                logger.warning(
                    "Phase1 LLM returned empty response on attempt %d/%d; retrying.",
                    idx + 1, len(attempts),
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Phase1 LLM call timed out after %.0fs on attempt %d/%d.",
                    self.request_timeout, idx + 1, len(attempts),
                )
            except Exception as exc:
                logger.warning(
                    "Phase1 LLM call failed on attempt %d/%d: %s",
                    idx + 1, len(attempts), exc,
                )
        logger.warning(
            "Phase1 LLM call exhausted all %d attempts; returning empty "
            "response (downstream singleton-rescue will take over).",
            len(attempts),
        )
        return ""

    async def cluster_triggers(
        self, stage_b_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run Phase 1 clustering on a single trajectory."""
        triggers = stage_b_payload.get("step_triggers") or []
        agent_framework_desc = str(stage_b_payload.get("agent_framework_description", "") or "")

        # Per-file token usage accumulator
        usage_acc: Dict[str, int] = {"input_tokens": 0, "reasoning_tokens": 0, "output_tokens": 0}

        if not triggers:
            return {
                **_echo_payload_fields(stage_b_payload),
                "instances": [],
                "token_usage": usage_acc,
            }

        # Add index to each trigger for the LLM to reference
        indexed_triggers = []
        for i, t in enumerate(triggers):
            indexed_triggers.append({"index": i, **t})

        trajectory_overview = _build_trajectory_overview(stage_b_payload)

        # Load the three-tier compression pool saved upstream so we can
        # feed the LLM per-step content (th1 for triggered steps, th3 for
        # the rest) when it clusters triggers into instances.
        step_pool = load_step_compressions_from_payload(stage_b_payload)
        try:
            total_steps_val = int(stage_b_payload.get("total_steps") or 0)
        except Exception:
            total_steps_val = 0
        trajectory_block = _build_trajectory_block(
            triggers=triggers,
            step_pool=step_pool,
            total_steps=total_steps_val,
        )

        user_prompt = _build_user_prompt(
            agent_framework_description=agent_framework_desc,
            trajectory_overview=trajectory_overview,
            stage_b_triggers=indexed_triggers,
            trajectory_block=trajectory_block,
        )

        response = await self._call_llm(PHASE1_SYSTEM_PROMPT, user_prompt, usage_acc=usage_acc)
        instances = _parse_phase1_response(response, triggers)

        return {
            **_echo_payload_fields(stage_b_payload),
            "instances": [inst.to_dict() for inst in instances],
            "token_usage": usage_acc,
        }

    async def process_file(
        self,
        stage_b_file: str,
        output_dir: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            with open(stage_b_file, "r", encoding="utf-8") as fh:
                stage_b_payload = json.load(fh)
        except Exception as exc:
            logger.error("Failed to load Stage B file %s: %s", stage_b_file, exc)
            return None

        if stage_b_payload.get("task_success"):
            payload = {
                **_echo_payload_fields(stage_b_payload),
                "instances": [],
                "token_usage": {"input_tokens": 0, "reasoning_tokens": 0, "output_tokens": 0},
            }
        else:
            payload = await self.cluster_triggers(stage_b_payload)

        stem = Path(stage_b_file).stem.replace("_stage_b", "")
        out_fp = Path(output_dir) / f"{stem}_stage_c_phase1.json"
        with out_fp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        n_instances = len(payload.get("instances") or [])
        token_usage = payload.get("token_usage") or {}
        logger.info(
            "Phase 1 [%s]: %d instances from %d triggers (tokens: in=%d, reason=%d, out=%d)",
            stem, n_instances, len(stage_b_payload.get("step_triggers") or []),
            token_usage.get("input_tokens", 0),
            token_usage.get("reasoning_tokens", 0),
            token_usage.get("output_tokens", 0),
        )
        return payload


def _echo_payload_fields(stage_b_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Echo fields from Stage B payload needed by Phase 2/3."""
    return {
        "task_id": stage_b_payload.get("task_id"),
        "task_description": stage_b_payload.get("task_description"),
        "task_success": stage_b_payload.get("task_success"),
        "task_outcome": stage_b_payload.get("task_outcome"),
        "reward": stage_b_payload.get("reward"),
        "environment": stage_b_payload.get("environment"),
        "total_steps": stage_b_payload.get("total_steps"),
        "trajectory_source": stage_b_payload.get("trajectory_source"),
        "agent_framework_description": stage_b_payload.get("agent_framework_description"),
        "step_triggers": stage_b_payload.get("step_triggers") or [],
        "step_compressions": stage_b_payload.get("step_compressions") or {},
        "trajectory_file_path": stage_b_payload.get("trajectory_file_path") or "",
        "metadata": stage_b_payload.get("metadata") or {},
    }


# =====================================================================
# Batch runner
# =====================================================================


def _is_valid_phase1_output(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return False
    return isinstance(data, dict) and "instances" in data


def _ensure_chat_completions_url(url: str) -> str:
    url = url.rstrip("/")
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url


_STAGE_B_SUFFIX = "_stage_b.json"


def _collect_stage_b_files(path: str) -> List[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        return [str(x) for x in sorted(p.rglob(f"*{_STAGE_B_SUFFIX}"))]
    raise FileNotFoundError(f"Path not found: {path}")


async def run_batch(
    stage_b_inputs: List[str],
    output_dir: str,
    api_config: Dict[str, Any],
    concurrency: int = 4,
    resume: bool = False,
    overwrite: bool = False,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    api_config = dict(api_config)
    api_config["base_url"] = _ensure_chat_completions_url(api_config["base_url"])

    clusterer = Phase1Clusterer(api_config)
    file_sem = asyncio.Semaphore(max(1, int(concurrency)))
    total = len(stage_b_inputs)

    async def _process_one(idx: int, sb_file: str) -> Tuple[str, str, Optional[str]]:
        stem = Path(sb_file).stem.replace("_stage_b", "")
        out_fp = Path(output_dir) / f"{stem}_stage_c_phase1.json"
        if overwrite:
            pass
        elif resume and _is_valid_phase1_output(out_fp):
            return ("skipped", sb_file, f"valid cached: {out_fp.name}")
        async with file_sem:
            logger.info("Phase 1 (%d/%d): %s", idx, total, sb_file)
            try:
                r = await clusterer.process_file(sb_file, output_dir)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                return ("failed", sb_file, repr(exc))
        if r is None:
            return ("failed", sb_file, "process_file returned None")
        return ("ok", sb_file, None)

    ok = failed = 0
    skipped_list: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    try:
        async_tasks = [_process_one(i, sb) for i, sb in enumerate(stage_b_inputs, start=1)]
        results = await asyncio.gather(*async_tasks)
        for status, sb, info in results:
            if status == "ok":
                ok += 1
            elif status == "failed":
                failed += 1
                failures.append({"stage_b_file": sb, "error": info or ""})
            else:
                skipped_list.append({"stage_b_file": sb, "reason": info or ""})
    finally:
        clusterer.close()

    summary = {
        "output_dir": output_dir,
        "num_inputs": len(stage_b_inputs),
        "num_ok": ok,
        "num_failed": failed,
        "num_skipped": len(skipped_list),
        "concurrency": int(concurrency),
        "resume": resume,
        "overwrite": overwrite,
        "skipped": skipped_list[:20],
        "failures": failures[:20],
    }
    logger.info("Phase 1 batch done: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage C Phase 1 (v5): Trigger clustering into error instances",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage_b_file", help="Single Stage B result json")
    group.add_argument("--stage_b_dir", help="Directory of Stage B results")

    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--api_key", default=os.getenv("API_KEY"))
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--judgement_force_json", type=lambda v: v.lower() not in ("0", "false", "no"), default=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--llm_concurrency",
        type=int,
        default=192,
        help=(
            "Global cap on in-flight LLM calls. Should be >= the "
            "router's MAX_CONCURRENT_REQUESTS so the router queue "
            "stays warm."
        ),
    )
    parser.add_argument(
        "--request_timeout",
        type=float,
        default=180.0,
        help="Wall-clock timeout (seconds) per LLM call.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--model_profile",
        default=None,
        help=(
            "Optional model profile name from utils/model_profiles.py. "
            "When set, the profile's `params` (excluding temperature / "
            "max_tokens which stay owned by CLI) are forwarded as extra "
            "OpenAI-compatible request parameters."
        ),
    )

    args = parser.parse_args()
    if not args.api_key:
        raise ValueError("Missing --api_key.")

    api_config = {
        "api_key": args.api_key,
        "base_url": args.base_url,
        "model": args.model,
        "temperature": args.temperature,
        "cache_url": args.cache,
        "max_tokens": args.max_tokens,
        "judgement_force_json": args.judgement_force_json,
        "model_profile": args.model_profile,
        "llm_concurrency": args.llm_concurrency,
        "request_timeout": args.request_timeout,
    }

    if args.stage_b_file:
        inputs = [args.stage_b_file]
    else:
        inputs = _collect_stage_b_files(args.stage_b_dir)

    summary = asyncio.run(
        run_batch(
            stage_b_inputs=inputs,
            output_dir=args.output_dir,
            api_config=api_config,
            concurrency=args.concurrency,
            resume=args.resume,
            overwrite=args.overwrite,
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()