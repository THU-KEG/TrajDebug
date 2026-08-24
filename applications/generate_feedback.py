#!/usr/bin/env python3
"""Generate actionable feedback from TRAJDEBUG diagnoses (paper Section 6).

Consumes Phase 3 `*_final.json` outputs and the original unified
trajectory to generate a developer-facing analysis report for each
failed trajectory.

The report includes:
  * Critical error analysis (conflict type, taxonomy, quotes)
  * Fix suggestion (hint sentence for the agent prompt)
  * Alternative critical step candidates (top-2)
  * Error statistics (mechanical)
  * Error classification (cascade, fixed, dormant, etc.)
  * Error timeline
  * Narrative summary

Usage:
    python applications/generate_feedback.py \\
        --final_dir outputs/alfworld_final \\
        --trajectory_dir data/unified/alfworld \\
        --output_dir outputs/alfworld_report \\
        --base_url <endpoint> \\
        --model <model> \\
        --api_key <key> \\
        --cache .cache/trajdebug/exp_qwen_alfworld_report.pkl \\
        --concurrency 32 \\
        --resume
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Reuse the detector modules while keeping feedback outside the core pipeline.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / 'detector'))
from typing import Any, Dict, List, Optional, Tuple

from _stage_common import (
    extract_last_json_object,
    load_step_compressions_from_payload,
    resolve_extra_params,
)
from utils.error_definitions import ErrorDefinitionsLoader
from utils.llm_compression import clip_text_middle, strip_think_tags
from utils.model import APIModel
from utils.trajectory_utils import (
    collect_trajectory_sources,
    flatten_unified_trajectory,
    load_trajectory_source,
    output_stem_for_source,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# =====================================================================
# Constants
# =====================================================================

_PER_STEP_CHARS = 800  # max chars per step in rendered trajectory

_CATEGORY_DESCRIPTIONS = {
    "cat-1": "Conflicts with TASK requirements (task instructions, constraints, output format specifications)",
    "cat-2": "Conflicts with visible CONTEXT (environment observations, tool output, system feedback, prior environment-rendered facts)",
    "cat-3": "Internally INCONSISTENT — contradicts the agent's own prior reasoning, plan, reflection, or memory",
}

_CATEGORY_DEFINITIONS_BLOCK = """cat-1 = the step conflicts with the TASK (task requirements, constraints, output format).
  The agent does something that directly violates what the task statement says to do.

cat-2 = the step conflicts with VISIBLE CONTEXT (environment observations, tool output, system feedback, prior environment-rendered facts).
  The agent ignores, misreads, or contradicts information that was explicitly shown to it.

cat-3 = the step is INTERNALLY INCONSISTENT — its claim contradicts the agent's own prior reasoning, plan, reflection, or memory.
  The agent contradicts what it itself previously stated or decided."""

CRITICAL_ERROR_DEFINITION = """A "critical error step" is the EARLIEST error instance that remains ACTIVE (unrepaired) at the terminal step of a failed trajectory — the first error whose effects were never truly repaired, making it the root cause of the failure chain.

An error is "critical" if:
- It represents the ROOT CAUSE that made task success impossible
- It caused a cascade of subsequent errors
- The trajectory could have succeeded if THIS specific error had not occurred
- Correcting this specific error would fundamentally change the trajectory toward success"""


# =====================================================================
# System / User prompt templates
# =====================================================================

REPORT_SYSTEM_PROMPT = (
    "You are an expert at analyzing failed agent trajectories and providing "
    "actionable debugging insights for developers. "
    "Respond with ONLY one valid JSON object that matches the requested schema."
)


def _build_taxonomy_block() -> str:
    """Build the full 25-tag taxonomy block (same format as direct baseline)."""
    loader = ErrorDefinitionsLoader()
    lines: List[str] = []
    for module_name, module_data in loader.definitions.items():
        for subtype, details in module_data.get("errors", {}).items():
            tag = f"{module_name}.{subtype}"
            definition = details.get("definition", "")
            if len(definition) > 320:
                definition = definition[:317] + "..."
            lines.append(f"- {tag}: {definition}")
    return "\n".join(lines)


def _render_clean_trajectory(
    flat_steps: List[Dict[str, Any]],
    step_pool: Optional[Dict[int, Dict[str, str]]],
    per_step_chars: int = _PER_STEP_CHARS,
) -> str:
    """Render trajectory WITHOUT trigger annotations (same as Phase 3)."""
    parts: List[str] = []
    for fs in flat_steps:
        step = fs.get("step", "?")
        role = fs.get("role", "assistant")
        body = ""
        pool = step_pool.get(step) if step_pool and isinstance(step, int) else None
        if pool:
            body = pool.get("th1") or pool.get("th2") or pool.get("th3") or ""
        if not body:
            body = str(fs.get("content", "") or "")
        body = clip_text_middle(body, per_step_chars).rstrip()
        header = f"Step {step} [role: {role}]:"
        if body.startswith(f"Step {step} [role:"):
            parts.append(body)
        else:
            parts.append(f"{header}\n{body}")
    return "\n\n".join(parts)


# =====================================================================
# Mechanical statistics computation
# =====================================================================


def _compute_statistics(final_data: Dict[str, Any]) -> Dict[str, Any]:
    """Compute error statistics from final.json (no LLM needed)."""
    instances = final_data.get("full_instance_states") or []
    summary = final_data.get("chain_summary") or {}
    total_steps = final_data.get("total_steps") or 1

    # Deduplicate wasted steps across instances (same step can appear in multiple)
    all_wasted_steps: set = set()
    for inst in instances:
        re_obj = inst.get("resource_effect") or {}
        if isinstance(re_obj, dict):
            ws = re_obj.get("wasted_steps") or []
            if isinstance(ws, list):
                all_wasted_steps.update(ws)
    total_wasted = len(all_wasted_steps)

    return {
        "total_instances": summary.get("num_total_instances", len(instances)),
        "fixed_instances": summary.get("num_fixed", 0),
        "active_instances": (summary.get("num_agent_active", 0)
                            + summary.get("num_env_active", 0)),
        "dormant_instances": summary.get("num_dormant", 0),
        "chain_instances": (summary.get("num_agent_chain", 0)
                           + summary.get("num_env_chain", 0)),
        "exploration_suppressed": summary.get("num_exploration_suppressed", 0),
        "span_cap_rescued": summary.get("num_span_cap", 0),
        "parse_failed": summary.get("num_parse_failed", 0),
        "total_wasted_steps": total_wasted,
        "wasted_step_ratio": round(total_wasted / max(total_steps, 1), 2),
    }


# =====================================================================
# Error timeline construction
# =====================================================================


def _build_error_timeline(final_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build a chronological error timeline from full_instance_states."""
    instances = final_data.get("full_instance_states") or []
    timeline: List[Dict[str, Any]] = []

    for inst in instances:
        re_obj = inst.get("resource_effect") or {}
        wasted = 0
        if isinstance(re_obj, dict):
            try:
                wasted = int(re_obj.get("wasted_step_count") or 0)
            except (TypeError, ValueError):
                pass

        fix_status = inst.get("fix_status", "active")
        state = inst.get("state")
        chain_membership = bool(inst.get("chain_membership"))
        exploration_suppressed = bool(inst.get("exploration_suppressed"))

        # Determine classification
        if exploration_suppressed:
            classification = "suppressed"
        elif fix_status and fix_status.startswith("fixed_at_step_"):
            classification = "fixed"
        elif state == "dormant" and not chain_membership:
            classification = "dormant"
        elif chain_membership:
            classification = "chain_member"
        else:
            classification = "active_non_chain"

        timeline.append({
            "instance_id": inst.get("instance_id"),
            "origin_step": inst.get("origin_step"),
            "qualified_origin_step": inst.get("qualified_origin_step"),
            "category": inst.get("category"),
            "taxonomy_tag": inst.get("qualified_origin_taxonomy_tag"),
            "error_content": inst.get("error_content", ""),
            "fix_status": fix_status,
            "state": state,
            "wasted_steps": wasted,
            "chain_membership": chain_membership,
            "terminal_connection": inst.get("terminal_connection"),
            "classification": classification,
        })

    # Sort by origin_step
    timeline.sort(key=lambda x: x.get("origin_step") or 0)
    return timeline


# =====================================================================
# Reference gathering for critical step
# =====================================================================


def _gather_critical_step_reference(final_data: Dict[str, Any]) -> Dict[str, Any]:
    """Gather upstream reference info for the critical step."""
    ce = final_data.get("critical_error") or {}
    crit_step = ce.get("unified_critical_step")
    if crit_step is None:
        return {"has_trigger": False, "trigger": None, "instance": None}

    # Try to find trigger(s) at critical step
    triggers_at_step = [
        t for t in (final_data.get("step_triggers") or [])
        if t.get("step") == crit_step
    ]

    if triggers_at_step:
        # Match by taxonomy_tag if multiple triggers at same step
        target_tag = ce.get("unified_critical_taxonomy_tag")
        matched = [t for t in triggers_at_step if t.get("taxonomy_tag") == target_tag]
        trigger = matched[0] if matched else triggers_at_step[0]
        return {"has_trigger": True, "trigger": trigger, "instance": None}

    # No trigger — find the supporting instance
    inst_id = ce.get("llm_pick_supporting_instance_id")
    # Also try matching by chain_min
    instance_info = None
    for inst in (final_data.get("full_instance_states") or []):
        if inst_id is not None and inst.get("instance_id") == inst_id:
            instance_info = inst
            break
        # Fallback: match by qualified_origin_step
        if inst.get("qualified_origin_step") == crit_step and inst.get("chain_membership"):
            instance_info = inst

    return {"has_trigger": False, "trigger": None, "instance": instance_info}


# =====================================================================
# Chain member summary for cascade analysis
# =====================================================================


def _render_chain_members_block(final_data: Dict[str, Any]) -> str:
    """Render chain member instances as a compact block for the LLM."""
    instances = final_data.get("full_instance_states") or []
    chain_members = [
        inst for inst in instances
        if inst.get("chain_membership") and not inst.get("exploration_suppressed")
    ]

    if not chain_members:
        return "(No chain member instances found.)"

    lines: List[str] = []
    for inst in sorted(chain_members, key=lambda x: x.get("origin_step") or 0):
        iid = inst.get("instance_id", "?")
        origin = inst.get("origin_step", "?")
        category = inst.get("category", "?")
        error_content = (inst.get("error_content") or "")[:200]
        chain_exp = (inst.get("chain_explanation") or "")[:200]
        tc = inst.get("terminal_connection", "?")
        lines.append(
            f"  Instance #{iid} (origin_step={origin}, category={category}, "
            f"terminal_connection={tc})"
        )
        if error_content:
            lines.append(f"    error: {error_content}")
        if chain_exp:
            lines.append(f"    chain_explanation: {chain_exp}")
        lines.append("")

    return "\n".join(lines)


# =====================================================================
# Prompt construction
# =====================================================================


def _build_report_prompt(
    final_data: Dict[str, Any],
    flat_steps: List[Dict[str, Any]],
    step_pool: Optional[Dict[int, Dict[str, str]]],
    stats: Dict[str, Any],
    reference: Dict[str, Any],
    taxonomy_block: str,
) -> List[Dict[str, str]]:
    """Build the full prompt for the report LLM call."""
    ce = final_data.get("critical_error") or {}
    metadata = final_data.get("metadata") or {}
    task_desc = metadata.get("task_description") or final_data.get("task_description", "")
    agent_fw = (metadata.get("extra") or {}).get("agent_framework_description", "")
    crit_step = ce.get("unified_critical_step", "?")
    chain_explanation = (
        ce.get("agent_chain_explanation")
        or ce.get("env_chain_explanation")
        or ""
    )

    # Render trajectory
    rendered_traj = _render_clean_trajectory(flat_steps, step_pool)

    # Build user prompt sections
    user_parts: List[str] = []

    # 1. Task
    user_parts.append(f"=== TASK ===\n{task_desc}")

    # 2. Agent framework (if available)
    if agent_fw:
        user_parts.append(f"=== AGENT FRAMEWORK ===\n{agent_fw}")

    # 3. Trajectory outcome
    last_step = flat_steps[-1].get("step", 0) if flat_steps else 0
    user_parts.append(
        f"=== TRAJECTORY OUTCOME ===\n"
        f"FAILED (reward=0). Total steps: {final_data.get('total_steps', last_step)}. "
        f"Last step index: {last_step}."
    )

    # 4. Full trajectory
    user_parts.append(f"=== FULL TRAJECTORY ===\n{rendered_traj}")

    # 5. Category definitions
    user_parts.append(f"=== CONFLICT CATEGORY DEFINITIONS ===\n{_CATEGORY_DEFINITIONS_BLOCK}")

    # 6. Taxonomy definitions
    user_parts.append(f"=== TAXONOMY DEFINITIONS (25 tags: <module>.<subtype>) ===\n{taxonomy_block}")

    # 7. Critical error definition
    user_parts.append(f"=== CRITICAL ERROR DEFINITION ===\n{CRITICAL_ERROR_DEFINITION}")

    # 8. Critical step info
    critical_section = (
        f"=== CRITICAL STEP (determined by upstream pipeline) ===\n"
        f"The critical error step is Step {crit_step}.\n"
    )
    # Add instance-level info if available
    inst_info = reference.get("instance")
    if inst_info:
        error_content = inst_info.get("error_content", "")
        what_violated = inst_info.get("what_is_being_violated", "")
        if error_content:
            critical_section += f"Error content: {error_content}\n"
        if what_violated:
            critical_section += f"What is being violated: {what_violated}\n"
    if chain_explanation:
        critical_section += f"Chain explanation: {chain_explanation}\n"
    user_parts.append(critical_section.rstrip())

    # 9. Upstream reference (conditional)
    if reference.get("has_trigger"):
        trigger = reference["trigger"]
        ref_section = (
            f"=== UPSTREAM REFERENCE (from Stage B per-step analysis) ===\n"
            f"The upstream pipeline detected the following at Step {crit_step}:\n"
            f"  - category: {trigger.get('category', '?')}\n"
            f"  - taxonomy_tag: {trigger.get('taxonomy_tag', '?')}\n"
            f"  - wrong_content_quote: \"{trigger.get('wrong_content_quote', '')}\"\n"
            f"  - reference_quote: \"{trigger.get('reference_quote', '')}\"\n"
            f"  - confidence_reasoning: {trigger.get('confidence_reasoning', '')}\n\n"
            f"Use this as reference; confirm or revise based on the full trajectory."
        )
    else:
        ref_section = (
            f"=== UPSTREAM REFERENCE ===\n"
            f"No upstream per-step annotation is available for Step {crit_step}. "
            f"Please analyze the trajectory to determine the conflict type, "
            f"conflicting quotes, and taxonomy classification for this step."
        )
    user_parts.append(ref_section)

    # 10. Error statistics
    stats_section = (
        f"=== ERROR STATISTICS (for context) ===\n"
        f"- Total error instances detected: {stats['total_instances']}\n"
        f"- Fixed during trajectory (self-corrected): {stats['fixed_instances']}\n"
        f"- Active (unfixed) at failure: {stats['active_instances']}\n"
        f"- In failure chain (causally connected): {stats['chain_instances']}\n"
        f"- Total wasted steps: {stats['total_wasted_steps']} / "
        f"{final_data.get('total_steps', '?')} "
        f"({int(stats['wasted_step_ratio'] * 100)}%)"
    )
    user_parts.append(stats_section)

    # 11. Chain member instances (for cascade analysis)
    chain_block = _render_chain_members_block(final_data)
    user_parts.append(
        f"=== CHAIN MEMBER INSTANCES (for cascade analysis) ===\n{chain_block}"
    )

    # 12. Output instructions
    output_section = (
        "=== WHAT TO OUTPUT ===\n"
        "Output ONE JSON object with the following keys:\n\n"
        "{\n"
        "  \"conflict_type\": \"cat-1\" | \"cat-2\" | \"cat-3\",\n"
        "  \"conflict_type_reason\": \"<1 sentence explaining why this category>\",\n"
        "  \"wrong_content_quote\": \"<verbatim quote from the trajectory at the critical step showing the error>\",\n"
        "  \"reference_quote\": \"<verbatim quote of the reference being violated (from task/context/prior reasoning)>\",\n"
        "  \"taxonomy_tag\": \"<module>.<subtype> from the taxonomy above>\",\n"
        "  \"taxonomy_reason\": \"<1 sentence explaining why this tag>\",\n"
        "  \"error_explanation\": \"<2-4 sentences explaining what went wrong, for a developer>\",\n"
        "  \"hint_sentence\": \"<single actionable instruction to add to the agent's system prompt to prevent this specific error>\",\n"
        "  \"alternative_critical_steps\": [\n"
        "    {\n"
        "      \"step\": <int>,\n"
        "      \"taxonomy_tag\": \"<module>.<subtype>\",\n"
        "      \"reason\": \"<1-2 sentences explaining why this step is also a strong root-cause candidate>\",\n"
        "      \"hint_sentence\": \"<single actionable instruction to prevent this specific alternative error>\"\n"
        "    }\n"
        "  ],\n"
        "  \"cascade_analysis\": [\n"
        "    {\"instance_id\": <int>, \"is_cascade\": true|false, \"reason\": \"<brief>\"}\n"
        "  ],\n"
        "  \"narrative_summary\": \"<one paragraph summarizing the full trajectory failure story>\"\n"
        "}\n\n"
        "IMPORTANT NOTES:\n"
        "- conflict_type: Classify the critical step's error according to the CONFLICT CATEGORY DEFINITIONS above.\n"
        "- wrong_content_quote and reference_quote: Must be VERBATIM substrings from the trajectory.\n"
        "- taxonomy_tag: Must be one of the 25 tags listed in TAXONOMY DEFINITIONS.\n"
        "- hint_sentence: A concise, actionable instruction that could be prepended to the agent's prompt to prevent this class of error.\n"
        "- alternative_critical_steps: Give your TOP-2 alternative choices for the critical error step "
        "(steps other than the current critical step that could also be considered root causes). "
        "Must be non-user steps. Rank by priority (strongest alternative first). "
        "Each alternative must include a hint_sentence specific to that error. "
        "If only one obvious error exists, return an empty array [].\n"
        "- cascade_analysis: For each CHAIN MEMBER INSTANCE listed above, determine whether it is a "
        "CASCADE effect of the critical error (the critical error caused or enabled this instance) "
        "or an INDEPENDENT error (would have occurred regardless of the critical error).\n"
        "- narrative_summary: Write a single paragraph summarizing the trajectory failure from start to end.\n\n"
        "Return ONLY the JSON object, no other text."
    )
    user_parts.append(output_section)

    user_content = "\n\n".join(user_parts)

    return [
        {"role": "system", "content": REPORT_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# =====================================================================
# LLM call + parse
# =====================================================================


def _parse_report_response(raw: str) -> Optional[Dict[str, Any]]:
    """Parse the LLM's JSON response."""
    raw = strip_think_tags(raw or "")
    data = extract_last_json_object(raw, must_have_key="conflict_type")
    if not isinstance(data, dict):
        data = extract_last_json_object(raw, must_have_key="taxonomy_tag")
    if not isinstance(data, dict):
        data = extract_last_json_object(raw)
    if isinstance(data, dict) and data.get("conflict_type"):
        return data
    return None


async def _llm_report_call(
    model: APIModel,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> Optional[Dict[str, Any]]:
    """Call the LLM with retry logic (up to 3 attempts).

    APIModel.generate_chat is synchronous, so we run it in a thread
    to avoid blocking the event loop.
    """
    attempts = [
        (max_tokens, temperature),
        (max_tokens, min(temperature + 0.3, 1.0)),
        (int(max_tokens * 1.5), min(temperature + 0.3, 1.0)),
    ]

    for attempt_idx, (mt, temp) in enumerate(attempts):
        try:
            raw, usage = await asyncio.to_thread(
                model.generate_chat,
                messages, mt, temp,
                {"type": "json_object"},
                True,  # return_usage
            )
        except Exception as exc:
            logger.warning("Report LLM call attempt %d failed: %s", attempt_idx + 1, exc)
            if attempt_idx < len(attempts) - 1:
                continue
            return None

        result = _parse_report_response(raw)
        if result is not None:
            return result

        logger.warning(
            "Report LLM parse failed on attempt %d (raw length=%d)",
            attempt_idx + 1, len(raw or ""),
        )

    return None


# =====================================================================
# Report assembly
# =====================================================================


def _classify_errors_with_cascade(
    final_data: Dict[str, Any],
    cascade_analysis: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Classify all instances into categories using mechanical rules + LLM cascade."""
    instances = final_data.get("full_instance_states") or []
    ce = final_data.get("critical_error") or {}

    # Build cascade lookup from LLM output
    cascade_map: Dict[int, bool] = {}
    for item in (cascade_analysis or []):
        try:
            iid = int(item.get("instance_id", -1))
            cascade_map[iid] = bool(item.get("is_cascade"))
        except (TypeError, ValueError):
            pass

    # Determine the critical instance ID
    critical_inst_id = ce.get("llm_pick_supporting_instance_id")
    if critical_inst_id is None:
        # Try to find by matching unified_critical_step
        crit_step = ce.get("unified_critical_step")
        for inst in instances:
            if (inst.get("qualified_origin_step") == crit_step
                    and inst.get("chain_membership")):
                critical_inst_id = inst.get("instance_id")
                break

    result: Dict[str, List[Dict[str, Any]]] = {
        "critical": [],
        "cascade": [],
        "independent_chain": [],
        "fixed": [],
        "dormant": [],
        "suppressed": [],
    }

    for inst in instances:
        iid = inst.get("instance_id")
        fix_status = inst.get("fix_status", "active")
        state = inst.get("state")
        chain_membership = bool(inst.get("chain_membership"))
        exploration_suppressed = bool(inst.get("exploration_suppressed"))

        compact = {
            "instance_id": iid,
            "origin_step": inst.get("origin_step"),
            "category": inst.get("category"),
            "error_content": (inst.get("error_content") or "")[:200],
        }

        if iid == critical_inst_id:
            result["critical"].append(compact)
        elif exploration_suppressed:
            result["suppressed"].append(compact)
        elif fix_status and str(fix_status).startswith("fixed_at_step_"):
            result["fixed"].append(compact)
        elif state == "dormant" and not chain_membership:
            result["dormant"].append(compact)
        elif chain_membership:
            # Use LLM cascade judgment
            is_cascade = cascade_map.get(iid)
            if is_cascade is True:
                result["cascade"].append(compact)
            else:
                result["independent_chain"].append(compact)
        else:
            # Active but not in chain (rare edge case)
            result["dormant"].append(compact)

    return result


def _assemble_report(
    final_data: Dict[str, Any],
    stats: Dict[str, Any],
    timeline: List[Dict[str, Any]],
    llm_output: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble the final report JSON."""
    ce = final_data.get("critical_error") or {}
    metadata = final_data.get("metadata") or {}

    # Cascade classification
    cascade_analysis = (llm_output or {}).get("cascade_analysis") or []
    error_classification = _classify_errors_with_cascade(final_data, cascade_analysis)

    # Update timeline with classification from error_classification
    classification_map: Dict[int, str] = {}
    for cls_name, items in error_classification.items():
        for item in items:
            classification_map[item.get("instance_id")] = cls_name
    for entry in timeline:
        iid = entry.get("instance_id")
        if iid in classification_map:
            entry["classification"] = classification_map[iid]

    report: Dict[str, Any] = {
        "task_id": final_data.get("task_id") or metadata.get("task_id", ""),
        "task_description": (
            metadata.get("task_description")
            or final_data.get("task_description", "")
        ),
        "task_outcome": "failure",
        "total_steps": final_data.get("total_steps", 0),
        "environment": metadata.get("dataset")
                       or final_data.get("environment", ""),
    }

    # Critical error analysis
    if llm_output:
        taxonomy_tag = llm_output.get("taxonomy_tag", "")
        module, subtype = "", ""
        if taxonomy_tag and "." in taxonomy_tag:
            module, subtype = taxonomy_tag.split(".", 1)

        report["critical_error_analysis"] = {
            "critical_step": ce.get("unified_critical_step"),
            "unified_source": ce.get("unified_source"),
            "conflict_type": llm_output.get("conflict_type"),
            "conflict_type_description": _CATEGORY_DESCRIPTIONS.get(
                llm_output.get("conflict_type", ""), ""
            ),
            "conflict_type_reason": llm_output.get("conflict_type_reason", ""),
            "wrong_content_quote": llm_output.get("wrong_content_quote", ""),
            "reference_quote": llm_output.get("reference_quote", ""),
            "taxonomy_tag": taxonomy_tag,
            "taxonomy_module": module,
            "taxonomy_subtype": subtype,
            "taxonomy_reason": llm_output.get("taxonomy_reason", ""),
            "error_explanation": llm_output.get("error_explanation", ""),
            "chain_explanation": (
                ce.get("agent_chain_explanation")
                or ce.get("env_chain_explanation")
                or ""
            ),
        }

        report["fix_suggestion"] = {
            "hint_sentence": llm_output.get("hint_sentence", ""),
            "narrative_summary": llm_output.get("narrative_summary", ""),
        }

        report["alternative_critical_steps"] = (
            llm_output.get("alternative_critical_steps") or []
        )
    else:
        # LLM failed — minimal report with Phase 3 data
        report["critical_error_analysis"] = {
            "critical_step": ce.get("unified_critical_step"),
            "unified_source": ce.get("unified_source"),
            "conflict_type": ce.get("agent_category"),
            "conflict_type_description": _CATEGORY_DESCRIPTIONS.get(
                ce.get("agent_category", ""), ""
            ),
            "conflict_type_reason": "",
            "wrong_content_quote": "",
            "reference_quote": "",
            "taxonomy_tag": ce.get("unified_critical_taxonomy_tag", ""),
            "taxonomy_module": ce.get("agent_affected_module", ""),
            "taxonomy_subtype": "",
            "taxonomy_reason": "",
            "error_explanation": ce.get("llm_pick_rationale", ""),
            "chain_explanation": (
                ce.get("agent_chain_explanation")
                or ce.get("env_chain_explanation")
                or ""
            ),
            "parse_failed": True,
        }
        report["fix_suggestion"] = {"hint_sentence": "", "narrative_summary": ""}
        report["alternative_critical_steps"] = []

    report["error_statistics"] = stats
    report["error_classification"] = error_classification
    report["error_timeline"] = timeline

    return report


# =====================================================================
# Per-file processing
# =====================================================================


async def _process_one(
    final_path: str,
    trajectory_dir: str,
    stage_a_dir: Optional[str],
    output_dir: str,
    model: APIModel,
    taxonomy_block: str,
    max_tokens: int,
    temperature: float,
    sem: asyncio.Semaphore,
) -> Optional[str]:
    """Process a single *_final.json file and generate its report."""
    async with sem:
        try:
            with open(final_path, "r", encoding="utf-8") as f:
                final_data = json.load(f)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", final_path, exc)
            return None

        # Skip successful trajectories
        reward = final_data.get("reward")
        if reward is None:
            reward = (final_data.get("metadata") or {}).get("reward")
        if reward == 1:
            logger.debug("Skipping successful trajectory: %s", final_path)
            return None

        # Skip if no critical error
        ce = final_data.get("critical_error") or {}
        if ce.get("unified_critical_step") is None:
            logger.debug("Skipping (no critical step): %s", final_path)
            return None

        # Determine output path
        stem = Path(final_path).stem
        # Remove _final suffix to get base stem
        if stem.endswith("_final"):
            base_stem = stem[:-6]
        else:
            base_stem = stem
        output_path = os.path.join(output_dir, f"{base_stem}_report.json")

        # Load unified trajectory for rendering
        traj_file = os.path.join(trajectory_dir, f"{base_stem}.json")
        flat_steps: List[Dict[str, Any]] = []
        step_pool: Optional[Dict[int, Dict[str, str]]] = None

        if os.path.isfile(traj_file):
            try:
                traj_data = load_trajectory_source(traj_file)
                flat_steps = flatten_unified_trajectory(traj_data)
            except Exception as exc:
                logger.warning("Failed to load trajectory %s: %s", traj_file, exc)

        # Try to load step compressions from Stage A output
        if stage_a_dir:
            stage_a_file = os.path.join(stage_a_dir, f"{base_stem}_stage_a.json")
            if os.path.isfile(stage_a_file):
                try:
                    with open(stage_a_file, "r", encoding="utf-8") as f:
                        stage_a_data = json.load(f)
                    step_pool = load_step_compressions_from_payload(stage_a_data)
                except Exception as exc:
                    logger.debug("Failed to load Stage A %s: %s", stage_a_file, exc)

        # If no flat_steps from trajectory, try to reconstruct from final_data
        if not flat_steps:
            logger.warning("No trajectory available for %s, prompt will lack trajectory", final_path)

        # Mechanical stats
        stats = _compute_statistics(final_data)
        timeline = _build_error_timeline(final_data)
        reference = _gather_critical_step_reference(final_data)

        # Build prompt
        messages = _build_report_prompt(
            final_data, flat_steps, step_pool, stats, reference, taxonomy_block
        )

        # LLM call
        llm_output = await _llm_report_call(model, messages, max_tokens, temperature)

        if llm_output is None:
            logger.warning("LLM report generation failed for %s", final_path)

        # Assemble report
        report = _assemble_report(final_data, stats, timeline, llm_output)

        # Write output
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info("Report written: %s", output_path)
        return output_path


# =====================================================================
# Batch runner
# =====================================================================


async def run_batch(
    final_dir: str,
    trajectory_dir: str,
    stage_a_dir: Optional[str],
    output_dir: str,
    model: APIModel,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    resume: bool,
) -> None:
    """Run report generation over all *_final.json files."""
    # Discover final files
    final_dir_path = Path(final_dir)
    if final_dir_path.is_file():
        final_files = [str(final_dir_path)]
    elif final_dir_path.is_dir():
        final_files = sorted(str(p) for p in final_dir_path.glob("*_final.json"))
    else:
        logger.error("final_dir not found: %s", final_dir)
        return

    if not final_files:
        logger.warning("No *_final.json files found in %s", final_dir)
        return

    logger.info("Found %d final files in %s", len(final_files), final_dir)

    # Build taxonomy block once
    taxonomy_block = _build_taxonomy_block()

    # Filter for resume
    if resume:
        os.makedirs(output_dir, exist_ok=True)
        existing = set(os.listdir(output_dir))
        filtered = []
        for fp in final_files:
            stem = Path(fp).stem
            base_stem = stem[:-6] if stem.endswith("_final") else stem
            report_name = f"{base_stem}_report.json"
            if report_name not in existing:
                filtered.append(fp)
            else:
                logger.debug("Skipping (resume): %s", fp)
        logger.info("After resume filter: %d files to process", len(filtered))
        final_files = filtered

    if not final_files:
        logger.info("Nothing to do (all files already processed).")
        return

    sem = asyncio.Semaphore(concurrency)
    tasks = [
        _process_one(
            fp, trajectory_dir, stage_a_dir, output_dir, model,
            taxonomy_block, max_tokens, temperature, sem,
        )
        for fp in final_files
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    success = sum(1 for r in results if isinstance(r, str))
    failures = sum(1 for r in results if r is None or isinstance(r, Exception))
    logger.info("Done: %d reports generated, %d skipped/failed", success, failures)


# =====================================================================
# CLI
# =====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate Section 6 feedback from TRAJDEBUG diagnoses."
    )
    parser.add_argument("--final_dir", required=True,
                        help="Directory of *_final.json files (or a single file path)")
    parser.add_argument("--trajectory_dir", required=True,
                        help="Directory of unified trajectory files")
    parser.add_argument("--stage_a_dir", default=None,
                        help="Directory of *_stage_a.json files (optional, for step compressions)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for *_report.json files")
    parser.add_argument("--base_url", required=True,
                        help="LLM API endpoint URL")
    parser.add_argument("--model", required=True,
                        help="Model name")
    parser.add_argument("--api_key", default="EMPTY",
                        help="API key (default: EMPTY)")
    parser.add_argument("--cache", default=".cache/trajdebug/feedback.pkl",
                        help="Path to pickle cache file")
    parser.add_argument("--concurrency", type=int, default=32,
                        help="Max concurrent files (default: 32)")
    parser.add_argument("--max_tokens", type=int, default=32768,
                        help="Max tokens for LLM response (default: 32768)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip files that already have reports")

    args = parser.parse_args()

    # Resolve extra params (backend, trust_env, etc.)
    extra = resolve_extra_params(None)

    # Initialize model
    # APIModel signature: (cache_url, base_url, model_name, api_key, extra_params)
    model = APIModel(
        args.cache,
        args.base_url,
        args.model,
        args.api_key,
        extra_params=extra,
    )

    asyncio.run(run_batch(
        final_dir=args.final_dir,
        trajectory_dir=args.trajectory_dir,
        stage_a_dir=args.stage_a_dir,
        output_dir=args.output_dir,
        model=model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        concurrency=args.concurrency,
        resume=args.resume,
    ))


if __name__ == "__main__":
    main()
