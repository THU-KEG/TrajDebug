#!/usr/bin/env python3
"""Stage C — Phase 2 (v5): Per-instance repair / state determination.

Per v5 spec §5.4.2 / §5.6.3, Phase 2 sends ONE LLM call per error
instance (parallelised), deciding whether each instance is:
  - fixed_at_step_<N>  (repaired and left the active set)
  - active / regular   (still active and recently touched)
  - active / dormant   (still active but lazily ignored)

Inputs:
  * ``<stem>_stage_c_phase1.json`` (instances + triggers + compressions)
  * the original unified trajectory file (for full rendering)

Output: ``<output_dir>/<stem>_stage_c_phase2.json``
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from _stage_common import (
    extract_last_json_object,
    load_step_compressions_from_payload,
    load_unified_for_stage,
    render_history_for_focus,
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

# Number of steps after last_trigger_step to render at th1 (configurable)
DEFAULT_POST_TRIGGER_TH1_WINDOW = 5

# Wall-clock timeout (seconds) for ONE LLM call within a Phase 2
# instance attempt. Reasoning models can run 60-90s normally; 180s
# gives ~3x safety margin without letting a stuck request block the
# global worker pool indefinitely.
DEFAULT_LLM_CALL_TIMEOUT = 180.0


# =====================================================================
# System prompt (v5 §5.6.3)
# =====================================================================

PHASE2_SYSTEM_PROMPT = r"""You are an instance-state annotator. You receive ONE error instance and the FULL trajectory. Phase 1 is high-recall and may flag exploratory or weakly committal steps; your job is to perform the commitment-strength recheck, the repair / state determination, and the terminal-connection audit defined in the methodology document.

=== KEY DEFINITIONS ===

  origin_step       = the step Phase 1 picked as the first trigger
                      for this instance. (May be exploratory.)
  qualified_origin_step
                    = the FIRST step in [origin_step, T] at which the
                      agent makes an OBSERVABLE WRONG COMMITMENT for
                      this instance's violated_object. May equal
                      origin_step, may be a later step, may be null.
  last_trigger_step = the latest step at which the same instance
                      re-fired in Phase 1.
  terminal step T   = the last message index in the trajectory.
  I (violated_object)
                    = the concrete object the instance is
                      violating (a TASK clause / a HISTORY object /
                      a same-message premise / a prior agent-self
                      statement).

=== TASK 1 — COMMITMENT-STRENGTH RECHECK ===

Phase 1 is high-recall: a trigger may fire at an early probe or
initial plan step even when the agent had no way yet to know the
action was wrong. Your job is to classify each origin_step into one
of three commitment strengths so that `qualified_origin_step`
reflects the FIRST step at which the agent observably persisted in
the wrong belief despite already-visible evidence.

A commitment to error requires BOTH:
  (a) the agent takes an observable binding action or states a
      non-hedged belief that contradicts the violated_object I, AND
  (b) by that step, the evidence contradicting I is already visible
      in the prior trajectory (previous observations, prior tool
      outputs, the same message's own premises, or the task
      statement itself).

An early plan or first tool call whose wrongness only becomes
knowable AFTER a later observation does NOT satisfy (b) at
origin_step, even if it turns out retrospectively wrong.

Classify the Phase 1 origin into ONE of:

  origin_commitment_status = "explicit_wrong_commitment"
    BOTH (a) and (b) hold at origin_step.
    → set qualified_origin_step = origin_step.
    Concretely:
      - cat-1: the task statement (always observable from step 0)
        names a constraint/format/deliverable, and at origin_step
        the agent emits or commits to something that directly
        violates it.
      - cat-2 / env: a specific prior observation or tool output
        (visible before origin_step) contradicts the agent's
        current action / claim, and that action is not hedged or
        retracted.
      - cat-3: the agent's own earlier plan / reflection / memory
        states one thing, and at origin_step the agent commits to
        an action that contradicts it without acknowledging the
        contradiction.

  origin_commitment_status = "weak_signal"
    EITHER (a) holds at origin_step but (b) does NOT (conflicting
    evidence not yet observable — origin_step is a first probe /
    first plan / first tool call whose output only later reveals
    it was wrong), OR (a) is only hedged ("let me try", "maybe",
    "first attempt") at origin_step.
    In both sub-cases there MUST be a clearly LATER step M
    (origin_step < M <= T) at which the agent, now able to see the
    conflicting evidence, still takes a binding action that
    contradicts I (or repeats a disconfirmed action).
    → set qualified_origin_step = M, set origin_relocated = true.

  origin_commitment_status = "pure_exploration"
    origin_step is plain exploration / setup / a reasonable first
    guess any reasonable agent might try, AND no step in
    (origin_step, T] satisfies the weak_signal relocation threshold
    (the agent never commits to violating I after the evidence
    becomes observable).
    → set qualified_origin_step = null, set exploration_suppressed =
      true.

RELOCATION GUIDE for weak_signal. To find M:
  1. Find the FIRST step M > origin_step where the agent has
     seen the contradicting evidence (an observation, a tool
     output, or a prior state the agent itself logged).
  2. Verify that at M the agent still takes an observable binding
     action / states a non-hedged belief that contradicts I.
     Hedged or retracting behavior does NOT count.
  3. If multiple such M exist, pick the earliest.
  4. If the only same-instance re-fires in (origin_step, T] are
     all weak/hedged, classify origin as pure_exploration.

TRIGGER-BASED HINT. If Phase 1 emitted multiple `trigger_indices`
for this instance, the first trigger at or after the step where the
contradicting evidence first appears is a reasonable default M,
unless the trajectory makes a later step a better fit.

SINGLE-TRIGGER SHORTCUT. If the instance has exactly ONE trigger
(origin_step == last_trigger_step) AND the instance is cat-2 or
cat-3 (NOT cat-1 or env), the default label is `pure_exploration`,
UNLESS a verbatim quote from a step strictly after origin_step shows
the agent referencing, repeating, or committing to the same wrong
object. Cat-1 and env single-trigger instances still follow the
HARD rules below.

  HARD CAT-1 RULES (two simple guarantees, mechanically enforced):
    1. cat-1 instance MUST NOT have origin_commitment_status =
       pure_exploration. The unsatisfied TASK clause itself is the
       wrong commitment; pick a non-null qualified_origin_step. For
       cat-1, the task statement is always "observable" by the agent
       from step 0, so condition (b) is trivially satisfied at any
       step where the agent takes a binding action that contradicts
       the task clause. The RELOCATION GUIDE still applies — prefer
       the first step where the agent actually binds to an
       incompatible action.
    2. cat-1 instance MUST NOT have state = dormant. cat-1 is by
       definition still violating a TASK clause at T, which is an
       observable terminal impact, not a dormant condition.

  HARD ENV RULES (symmetric to cat-1, mechanically enforced):
    1. An env instance MUST NOT have origin_commitment_status =
       pure_exploration UNLESS BOTH of the following hold:
         (a) the environment self-recovers on the next judgable step
             (returns a normal payload for the same sub-goal), AND
         (b) the agent's next action is consistent with the recovered
             output.
       Rationale: an env trigger is a violation that ALREADY HAPPENED
       in the environment output — it is not a tentative agent probe.
       `pure_exploration` judges the AGENT's commitment, not the
       environment's behavior, so the label almost never fits env.
       If neither (a) nor (b) holds, use explicit_wrong_commitment
       with qualified_origin_step = origin_step.
    2. For env instances, wasted steps should be counted against the
       VIOLATED sub-goal object remaining unresolved (the environment
       still refusing to return a usable payload, OR the agent still
       re-firing the same failing request / hitting the same hijack),
       NOT against whether the agent's intent was blameworthy. Each
       re-hit of the same env anomaly in (qualified_origin_step, T]
       and each step spent recovering from it counts as 1 wasted step.

=== TASK 2 — REPAIR / STATE DETERMINATION ===

Using qualified_origin_step (or origin_step if qualified is null) as
the anchor, decide one of THREE labels.

  fixed_at_step_<N>
    There is a step N with anchor < N <= T at which the agent's
    behavior satisfies the repair criterion for this instance's
    category. The instance leaves the active set.

  active / regular
    No step in (anchor, T] satisfies the repair criterion, AND at
    least one judgable step in (anchor, T] has an observable impact
    on I — re-fires the same error, continues using the wrong
    reference, repeatedly violates the same constraint, OR causes
    measurable budget debt.

  active / dormant
    No step in (anchor, T] satisfies the repair criterion, AND no
    judgable step in (anchor, T] has an observable impact on I — the
    agent neither fixed nor touched I again.
    **HARD: cat-1 instances can NEVER be dormant** (see HARD CAT-1
    RULES in TASK 1). If a cat-1 instance looks dormant, re-label it
    active/regular.
    **HARD: pure_exploration suppression** — if you set
    exploration_suppressed=true above, you MUST set fix_status=active,
    state=dormant here, regardless of any other signals. This is the
    §3.5 condition-7 path.

=== TASK 3 — TERMINAL-CONNECTION AUDIT (FAILURE-PATH TEST) ===

Decide whether THIS instance lies on the actual observable path
that took the trajectory to its terminal failure. This is NOT a
"single necessary cause" test — a failure often has multiple
co-occurring wrong commitments, and each can be on the path.

Anchor your judgment on the OBSERVED terminal state given at the
top of the user message (`task_success` + last 1-2 steps). Do NOT
imagine an alternative trajectory.

Run the two questions below in order.

  Q1 — FAILURE-PATH MEMBERSHIP.
       Does any of the following appear on the path that produced
       the observed terminal failure?
         (i)   the instance's wrong commitment is still active at T,
               OR re-fires after qualified_origin_step;
         (ii)  the instance is reflected in the terminal answer,
               terminal action, terminal hijacked / errored state;
         (iii) steps caused by this instance (re-fires, retries
               against the same wrong belief, repeated reads of an
               irrelevant resource, env re-hits at the same domain /
               same error code) materially consumed step-budget so
               the trajectory exhausted its budget before finishing;
         (iv)  this instance produced an irreversible state change
               (purchase / submit / external write) that blocks
               recovery and is visible at T.
       If ANY of (i)–(iv) holds → the instance IS on the failure path.

  Q2 — REPAIR-AND-MOVED-ON ESCAPE.
       Set terminal_connection = "none" ONLY if ALL of the following
       hold:
         (a) the violated_object was semantically repaired (fix_status
             = fixed_at_step_<N>, semantic_status = fixed); AND
         (b) wasted_step_count <= 5 (so the repair did not consume a
             material share of the step budget); AND
         (c) the trajectory's terminal failure is attributable to a
             clearly DIFFERENT cause that you can name (a different
             wrong commitment / different env anomaly / a non-
             instance factor such as the agent simply not knowing
             the answer).
       If you cannot name a different dominating cause under (c),
       you do NOT have a clean repair-and-moved-on escape — the
       instance stays on the failure path.

  MULTI-INSTANCE GUARD. Several instances on the same trajectory
  can each be on the failure path simultaneously. Do NOT eliminate
  this instance just because other instances also exist or look
  more salient. Judge this instance on its own evidence.

Decide terminal_connection, one of:

  "irreversible"    — Q1(iv) holds: the instance produced an
                      irreversible state change directly visible at
                      terminal step T (submitted wrong order,
                      purchased wrong item, written wrong file).
  "semantic"        — Q1(i) or Q1(ii) holds: the wrong commitment
                      is still active at T, or is visibly reflected
                      in the terminal answer / terminal action /
                      terminal failure state. Quote the strongest
                      evidence step.
  "budget_debt"     — Q1(iii) holds and neither (i) nor (ii)
                      cleanly applies: the instance's wasted steps
                      consumed a material share of the step budget.
                      Required precondition: wasted_step_count > 5.
                      If 1 <= wasted_step_count <= 5 and a semantic
                      carrier also exists, prefer "semantic".
  "none"            — Q2 fully passes (repaired AND wasted <= 5 AND
                      a clearly different dominating cause), or Q1
                      has no evidence at all (a truly local mistake
                      whose effect did not propagate).

When uncertain between "none" and a connected label, and the
instance has any post-qos re-fire or active fix_status, prefer the
connected label.

Then set chain_membership = (terminal_connection != "none").

If qualified_origin_step is null (pure_exploration suppressed),
force terminal_connection="none" and chain_membership=false.

EXPLANATION REQUIREMENT. In `chain_explanation` you MUST:
  1. State the trajectory's terminal failure mode in one phrase.
  2. State which of Q1(i)-(iv) holds, OR if you took the Q2 escape,
     name the different dominating cause (another instance id, or
     a non-instance factor).
  3. Quote the single strongest verbatim evidence step.

=== TASK 4 — RESOURCE EFFECT (fill whenever wasted steps exist) ===

Fill `resource_effect` whenever the instance wasted ANY steps in
(qualified_origin_step, T], REGARDLESS of fix_status. This includes
the "fixed-but-costly" case where the agent eventually repaired the
error but burned many steps before doing so — those steps are still
budget debt that connects the instance to the terminal failure under
§3.6.

Resource effect schema:
  {
    "wasted_steps": [<list of step indices wasted on I>],
    "wasted_step_count": <int>,
    "budget_debt": "none" | "possible" | "confirmed",
    "last_referencing_step": <int or null>
  }
Set `resource_effect = null` ONLY when wasted_step_count would be 0
AND state is not regular (e.g. dormant or fixed at the very next step
with no waste in between).

**MECHANICAL BUDGET-DEBT RULE.** This eliminates subjective judgment:
  - wasted_step_count == 0 -> budget_debt = "none"
  - 1 <= wasted_step_count <= 5 -> budget_debt = "possible"
  - wasted_step_count > 5 -> budget_debt = "confirmed"

Definition of `wasted_step_count`: the number of distinct steps in
(qualified_origin_step, fix_step or T] whose action / output is
judgably caused by this instance's wrong commitment AND is not
productive towards the repair criterion. For fixed instances, count
the wasted steps in (origin_step, fixed_at_step]; the repair step
itself is NOT wasted but the steps before it usually are. Re-firing
triggers, retries against the same wrong belief, repeated reads of an
irrelevant resource, or `agent does nothing useful` filler steps all
count.

**Coupling with terminal_connection for fixed-but-costly errors.**
If fix_status = fixed_at_step_<N> AND wasted_step_count > 5, you
SHOULD consider:
  - terminal_connection = "budget_debt"
  - chain_membership = true
  - state stays null (because fix_status is not active)
This is the only path through which a semantically-fixed instance can
still be in the error chain, and §3.6 explicitly endorses it.
HOWEVER, this is NOT automatic anymore: the strict root-cause test
in Task 3 still applies. Pick budget_debt ONLY if the wasted steps
caused by I plausibly account for the DECISIVE share of budget
exhaustion at T (i.e. the trajectory would have finished if I had
been absent). If some OTHER instance dominates the wasted budget,
or the trajectory failed for a non-budget reason (wrong final
answer, wrong submitted action) that is unrelated to I, then I is
NOT chain-connected even though wasted_step_count > 5; in that case
terminal_connection = "none" and chain_membership = false.

**Downstream override.** Phase 3 will independently re-derive
budget_debt from wasted_step_count using the rule above; if you
leave wasted_step_count unset, Phase 3 will default it to
`max(0, last_trigger_step - origin_step - 1)` so the legacy
span_exceeds_cap signal is folded into budget_debt. Provide the most
faithful wasted_step_count you can, but do not invent steps.

=== REPAIR CRITERIA (per category) ===

  cat-1 : at step N the agent touches the violated TASK clause AND
          the behavior is consistent with that constraint.
  cat-2 : at step N the agent touches the violated CONTEXT object
          AND the behavior is consistent with the latest visible
          state of that object.
  cat-3 : at step N the agent explicitly retracts or corrects the
          prior wrong claim (verbatim retraction or re-derivation).
  env   : (env-fix-1) the agent detects the env anomaly and switches
          to a MATERIALLY DIFFERENT strategy that no longer depends on
          the failing env path; OR (env-fix-2) the env recovers AND
          the agent's subsequent behavior is consistent with the
          recovered output.

  HARD ENV-FIX RULES (mechanically enforced, do NOT bypass):
    (R1) The following are NOT a strategy switch and therefore do NOT
         qualify as env-fix-1:
           - clicking the browser back button / navigating to the
             previous page;
           - retrying the same URL / same query / same tool call;
           - reloading or staying on the same hijacked / errored
             page;
           - issuing a new request that lands on the SAME violated
             object (same domain, same utm/gclid signature, same
             error-code surface, same hijacked landing page).
         These are continuations of the same env exposure, not a
         switch. If the agent only does these things, fix_status
         MUST stay "active" (regular if there is any later judgable
         step, dormant only if the agent never touches the violated
         object again).
    (R2) Same-source RE-FIRE invalidates a prior fix. If you would
         otherwise label fix_status = fixed_at_step_<N>, but the SAME
         env instance re-fires at any step M with M > N (the trigger
         hits the same violated_object again — same hijack domain,
         same utm/gclid, same error code, same failing tool), then
         the fix did NOT hold: set fix_status = "active",
         state = "regular", and pick state_evidence_quote from the
         re-fire step M. The `last_trigger_step` field on the
         instance header already encodes the latest re-fire; if
         last_trigger_step > N you are in this case.
    (R3) A genuine env-fix-1 requires BOTH (a) the agent verbally /
         in its action explicitly recognises the env anomaly (e.g.
         "this is an ad / hijack / wrong page, abandon it"), AND
         (b) the next observable action targets a DIFFERENT path
         (different domain / different tool / a clearly new query
         that is not a paraphrase of the failing one). If only (a)
         or only (b) is present, prefer fix_status = "active".

"Touches" = the subsequent step has an observable impact on the
constraint / object — referencing it, acting on it, or making
decisions visibly conditioned on it.

=== EVIDENCE REQUIREMENTS ===

  - fix_status = fixed_at_step_<N>:
      state                = null;
      fix_evidence_quote   = verbatim quote from step N showing the
                             repair behavior.
  - fix_status = active, state = regular:
      fix_evidence_quote   = null;
      state_evidence_quote = verbatim quote from one
                             "stepping-on-it-again" step in
                             (anchor, T].
  - fix_status = active, state = dormant:
      fix_evidence_quote   = null;
      state_evidence_quote = null.

A later step also being labelled wrong by Stage B is NOT by itself a
repair signal — judge by whether the agent's BEHAVIOR at that step
satisfies the repair criterion above.

=== OUTPUT FORMAT ===

REMINDER. Before you finalize the output, recheck:
  - Did I anchor my Task 3 judgment on the OBSERVED terminal state
    given at the top of the user prompt?
  - Did I run the failure-path test (Q1) and the repair-and-moved-on
    escape (Q2) in Task 3?
  - Did I avoid eliminating this instance just because some OTHER
    instance also looks plausible? (multi-instance guard)
  - Does my chain_explanation name (i)–(iv) for the connected case,
    or name a different dominating cause for the "none" case?
  - terminal_connection is one of: "irreversible", "semantic",
    "budget_debt", "none". Do NOT emit "semantic_uncertain".

Return STRICT JSON with EXACTLY these keys (extra keys are ignored,
missing keys default conservatively):

{
  "instance_id": <int, copy from input>,
  "origin_commitment_status": "explicit_wrong_commitment" | "weak_signal" | "pure_exploration",
  "qualified_origin_step": <int or null>,
  "qualified_origin_trigger_id": <int or null, the trigger_index (from the instance header) whose step equals qualified_origin_step; null if no trigger lands exactly on that step>,
  "qualified_origin_taxonomy_tag": "<taxonomy_tag string of the trigger at qualified_origin_step, or null>",
  "origin_relocated": <bool>,
  "exploration_suppressed": <bool>,
  "fix_status": "active" | "fixed_at_step_<N>",
  "fix_evidence_quote": "<verbatim quote or null>",
  "state": "regular" | "dormant" | null,
  "state_evidence_quote": "<verbatim quote or null>",
  "semantic_status": "fixed" | "active",
  "terminal_connection": "irreversible" | "semantic" | "budget_debt" | "none",
  "chain_membership": <bool>,
  "resource_effect": <object or null>,
  "chain_explanation": "<2-4 sentences. MUST include: (1) the trajectory's terminal failure mode in one phrase, (2) which of Q1(i)-(iv) holds OR the named dominating cause if Q2 escape was taken, (3) one verbatim evidence quote.>"
}
"""


# =====================================================================
# Trajectory + instance inline rendering helpers
# =====================================================================

_QUOTE_TRUNC = 240  # max chars per quote in the inline render


def _truncate(s: Any, n: int = _QUOTE_TRUNC) -> str:
    if s is None:
        return ""
    s = str(s).replace("\n", " ").strip()
    if len(s) <= n:
        return s
    return s[:n] + " ...(truncated)"


def _render_trajectory_with_instance_inline(
    flat_steps: List[Dict[str, Any]],
    step_pool: Dict[int, Dict[str, str]],
    instance: Dict[str, Any],
) -> str:
    """Render the full trajectory and inline instance annotation at
    origin_step and last_trigger_step.

    At origin_step: show the instance's error_content, category,
    violated_object, and sub_goal so the LLM can immediately see
    what Phase 1 flagged at that step.

    At last_trigger_step (if different from origin_step): show a
    brief re-fire note so the LLM knows the instance was still active
    at that step.
    """
    origin = int(instance.get("origin_step", 0))
    last_trigger = int(instance.get("last_trigger_step", origin))
    category = instance.get("category", "?")
    iid = int(instance.get("instance_id", -1))
    error_content = _truncate(instance.get("error_content") or "", 360)
    sub_goal = _truncate(instance.get("sub_goal") or "", 200)
    what_violated = _truncate(instance.get("what_is_being_violated") or "", 200)
    wrong_quote = _truncate(
        instance.get("wrong_content_quote") or instance.get("wrong_quote") or ""
    )
    ref_quote = _truncate(
        instance.get("reference_quote") or instance.get("ref_quote") or ""
    )

    parts: List[str] = []
    for fs in flat_steps:
        step = int(fs.get("step", -1))
        role = fs.get("role", "?")

        # Render step content: prefer th1 from step_pool, else raw content.
        body = ""
        pool = step_pool.get(step) if step_pool else None
        if pool:
            body = pool.get("th1") or pool.get("th2") or pool.get("th3") or ""
        if not body:
            body = str(fs.get("content", "") or "")

        parts.append(f"--- step={step} role={role} ---")
        parts.append(body.rstrip())

        # Inline instance annotation at origin_step.
        if step == origin:
            parts.append(f"  ⚑ INSTANCE #{iid} ORIGIN @ step {step} [{category}]:")
            if error_content:
                parts.append(f"      error_content:     {error_content}")
            if what_violated:
                parts.append(f"      what_violated:     {what_violated}")
            if sub_goal:
                parts.append(f"      sub_goal:          {sub_goal}")
            if wrong_quote:
                parts.append(f"      wrong_quote:       \"{wrong_quote}\"")
            if ref_quote:
                parts.append(f"      reference_quote:   \"{ref_quote}\"")

        # Inline re-fire note at last_trigger_step (if different from origin).
        elif step == last_trigger and last_trigger != origin:
            parts.append(
                f"  ⚑ INSTANCE #{iid} LAST TRIGGER @ step {step} [{category}] "
                f"(re-fired from origin_step={origin})"
            )
            if wrong_quote:
                parts.append(f"      wrong_quote:       \"{wrong_quote}\"")

        parts.append("")  # blank line between steps

    return "\n".join(parts)


def _render_instance_header(
    instance: Dict[str, Any],
    step_triggers: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Compact one-block summary of the instance placed BEFORE the
    trajectory so the LLM has the full picture up front.

    trigger_indices are shown in the header line, and the full trigger
    details (taxonomy_tag, wrong_content_quote, reference_quote,
    why_wrong) are listed below the instance fields so the LLM can
    identify which trigger corresponds to qualified_origin_step.
    """
    iid = int(instance.get("instance_id", -1))
    category = instance.get("category", "?")
    origin = instance.get("origin_step", "?")
    last = instance.get("last_trigger_step", "?")
    attribution = (instance.get("attribution") or "agent").upper()
    fix_status = instance.get("fix_status", "?")
    trigger_indices = instance.get("trigger_indices") or []
    sub_goal = _truncate(instance.get("sub_goal") or "", 200)
    error_content = _truncate(instance.get("error_content") or "", 360)
    what_violated = _truncate(instance.get("what_is_being_violated") or "", 200)
    wrong_quote = _truncate(
        instance.get("wrong_content_quote") or instance.get("wrong_quote") or ""
    )
    ref_quote = _truncate(
        instance.get("reference_quote") or instance.get("ref_quote") or ""
    )

    lines = [
        f"[Instance #{iid}] {attribution} / {category}  "
        f"(origin_step={origin}, last_trigger_step={last}, "
        f"trigger_indices={trigger_indices}, phase1_fix_status={fix_status})",
    ]
    if sub_goal:
        lines.append(f"  sub_goal:          {sub_goal}")
    if error_content:
        lines.append(f"  error_content:     {error_content}")
    if what_violated:
        lines.append(f"  what_violated:     {what_violated}")
    if wrong_quote:
        lines.append(f"  wrong_quote:       \"{wrong_quote}\"")
    if ref_quote:
        lines.append(f"  reference_quote:   \"{ref_quote}\"")

    # Show trigger details so the LLM can map trigger_index -> step
    # and pick qualified_origin_trigger_id / qualified_origin_taxonomy_tag.
    if step_triggers and trigger_indices:
        lines.append("  triggers:")
        for tidx in trigger_indices:
            if not isinstance(tidx, int) or tidx < 0 or tidx >= len(step_triggers):
                continue
            t = step_triggers[tidx]
            t_step = t.get("step", "?")
            t_tag = t.get("taxonomy_tag", "?")
            t_wq = _truncate(t.get("wrong_content_quote") or "", 200)
            t_rq = _truncate(t.get("reference_quote") or "", 200)
            t_why = _truncate(t.get("why_wrong") or "", 200)
            lines.append(f"    [trigger_index={tidx}] step={t_step}  taxonomy_tag={t_tag}")
            if t_wq:
                lines.append(f"      wrong_content_quote: \"{t_wq}\"")
            if t_rq:
                lines.append(f"      reference_quote:     \"{t_rq}\"")
            if t_why:
                lines.append(f"      why_wrong:           {t_why}")

    return "\n".join(lines)


# =====================================================================
# User prompt builder
# =====================================================================


def _render_trajectory_outcome(
    task_success: Optional[bool],
    task_outcome: Optional[str],
    flat_steps: List[Dict[str, Any]],
    step_pool: Dict[int, Dict[str, str]],
) -> str:
    """Build a compact summary of how the trajectory ended.

    Renders three things so the LLM can ground Task 3 on the OBSERVED
    terminal state rather than imagining an alternative:

      * task_success (true / false / unknown);
      * task_outcome string from the upstream payload (may name the
        terminal failure mode, e.g. "step_budget_exhausted");
      * up to the last 2 step bodies (th3 -> th2 -> th1 -> raw),
        truncated to keep prompt small.

    No inference is added here — the LLM should read these signals
    and decide the terminal failure mode itself.
    """
    lines: List[str] = []
    if task_success is True:
        lines.append("task_success: true")
    elif task_success is False:
        lines.append("task_success: false")
    else:
        lines.append("task_success: unknown")
    if task_outcome:
        outcome_str = str(task_outcome).strip().replace("\n", " ")
        if len(outcome_str) > 240:
            outcome_str = outcome_str[:240] + " ...(truncated)"
        lines.append(f"task_outcome: {outcome_str}")

    # Last 2 steps as a brief window (oldest first).
    if flat_steps:
        last_steps = flat_steps[-2:] if len(flat_steps) >= 2 else flat_steps
        lines.append("last_steps:")
        for fs in last_steps:
            try:
                step = int(fs.get("step", -1))
            except (TypeError, ValueError):
                step = -1
            role = fs.get("role", "?")
            body = ""
            pool = step_pool.get(step) if step_pool else None
            if pool:
                body = pool.get("th3") or pool.get("th2") or pool.get("th1") or ""
            if not body:
                body = str(fs.get("content", "") or "")
            body = body.replace("\n", " ").strip()
            if len(body) > 360:
                body = body[:360] + " ...(truncated)"
            lines.append(f"  step={step} role={role}: {body}")

    return "\n".join(lines)


def _build_user_prompt(
    agent_framework_description: str,
    instance: Dict[str, Any],
    rendered_trajectory: str,
    step_triggers: Optional[List[Dict[str, Any]]] = None,
    trajectory_outcome_block: Optional[str] = None,
) -> str:
    parts = []
    if agent_framework_description.strip():
        parts.append(f"=== AGENT FRAMEWORK ===\n{agent_framework_description.strip()}")
    if trajectory_outcome_block and trajectory_outcome_block.strip():
        parts.append(f"=== TRAJECTORY OUTCOME ===\n{trajectory_outcome_block.strip()}")
    parts.append(
        f"=== INSTANCE ===\n"
        f"{_render_instance_header(instance, step_triggers=step_triggers)}"
    )
    parts.append(f"=== TRAJECTORY (with inline instance annotations) ===\n{rendered_trajectory}")
    return "\n\n".join(parts)


# =====================================================================
# Parser
# =====================================================================


_VALID_COMMITMENT = ("explicit_wrong_commitment", "weak_signal", "pure_exploration")
_VALID_TERMINAL = (
    "irreversible",
    "semantic",
    "budget_debt",
    "semantic_uncertain",
    "none",
)


def _coerce_int_or_none(v: Any) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _coerce_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes")
    return default


def _parse_phase2_response(
    response: str,
    instance: Dict[str, Any],
) -> Dict[str, Any]:
    """Parse Phase 2 LLM output for one instance, with v6 schema.

    Returns ``None`` if the response is unparseable so that the caller
    can drive the 3-retry-with-perturbation loop. The conservative
    fallback (parse_failed=true / active+regular / semantic_uncertain)
    is produced by ``_default_result`` and is applied AFTER all
    retries are exhausted.
    """
    instance_id = int(instance.get("instance_id", 0))
    origin_step = _coerce_int_or_none(instance.get("origin_step")) or 0
    category = instance.get("category", "")

    response = strip_think_tags(response or "")
    if response.strip().startswith("```"):
        response = "\n".join(
            line for line in response.splitlines()
            if not line.strip().startswith("```")
        )
    data = extract_last_json_object(response, must_have_key="instance_id")
    if data is None:
        data = extract_last_json_object(response, must_have_key="fix_status")
    if data is None:
        data = extract_last_json_object(response)
    if not isinstance(data, dict):
        return None  # signals retry / fallback

    # ---- commitment recheck ----
    commit = str(data.get("origin_commitment_status", "")).strip()
    if commit not in _VALID_COMMITMENT:
        commit = "explicit_wrong_commitment"
    qos = _coerce_int_or_none(data.get("qualified_origin_step"))
    origin_relocated = _coerce_bool(data.get("origin_relocated"), False)
    exploration_suppressed = _coerce_bool(data.get("exploration_suppressed"), False)

    # cat-1 hard rule: pure_exploration is forbidden.
    if category == "cat-1" and commit == "pure_exploration":
        commit = "explicit_wrong_commitment"
        if qos is None:
            qos = origin_step
        exploration_suppressed = False

    # env hard rule (symmetric): an env trigger is an already-observed
    # environment violation, not a tentative agent probe. The only
    # legitimate pure_exploration path for env is
    # (env-recovers AND agent-consistent-on-next-step), which is
    # virtually always equivalent to fixed_at_step_<origin+1>. Any
    # env instance still marked pure_exploration at the LLM layer
    # is re-labeled explicit_wrong_commitment here; if the env truly
    # self-recovered the LLM should have emitted fix_status=
    # fixed_at_step_<N> in the first place, and the downstream
    # sanitize layer will additionally catch cases where the LLM
    # kept exploration_suppressed=true without a matching fix.
    if category == "env" and commit == "pure_exploration":
        commit = "explicit_wrong_commitment"
        if qos is None:
            qos = origin_step
        exploration_suppressed = False

    if commit == "explicit_wrong_commitment":
        if qos is None:
            qos = origin_step
        origin_relocated = qos != origin_step
        exploration_suppressed = False
    elif commit == "weak_signal":
        if qos is None or qos < origin_step:
            qos = origin_step
            origin_relocated = False
        else:
            origin_relocated = qos != origin_step
        exploration_suppressed = False
    else:  # pure_exploration
        qos = None
        origin_relocated = False
        exploration_suppressed = True

    # ---- fix_status / state ----
    fix_status = str(data.get("fix_status", "")).strip()
    if fix_status != "active" and not re.match(r"^fixed_at_step_\d+$", fix_status):
        fix_status = "active"
    state = data.get("state")
    if fix_status != "active":
        state = None
    elif state not in ("regular", "dormant"):
        state = "regular"

    # cat-1 cannot be dormant.
    if category == "cat-1" and fix_status == "active" and state == "dormant":
        state = "regular"

    # pure_exploration suppression: force active/dormant.
    if exploration_suppressed:
        fix_status = "active"
        state = "dormant"

    # ---- terminal connection / chain membership ----
    terminal_connection = str(data.get("terminal_connection", "")).strip()
    if terminal_connection not in _VALID_TERMINAL:
        # v6 strict policy: when the LLM omits / gives an invalid
        # terminal_connection, default to "none" (safe-out) rather
        # than "semantic_uncertain" (auto-include). The strict
        # root-cause test in Task 3 requires affirmative evidence;
        # absence of evidence is evidence of absence here.
        terminal_connection = "none"
    chain_membership = _coerce_bool(
        data.get("chain_membership"),
        terminal_connection != "none",
    )
    if exploration_suppressed:
        terminal_connection = "none"
        chain_membership = False
    # Enforce the (terminal_connection == "none") => chain_membership
    # is False invariant defensively. Even if the LLM contradicts
    # itself by saying chain_membership=true while terminal_connection
    # is "none", we trust terminal_connection (the more specific
    # signal) and zero out chain_membership.
    if terminal_connection == "none":
        chain_membership = False

    # ---- semantic_status ----
    semantic_status = str(data.get("semantic_status", "")).strip()
    if semantic_status not in ("fixed", "active"):
        semantic_status = "fixed" if fix_status != "active" else "active"

    # ---- resource_effect ----
    # v6.1: fixed-but-costly errors must KEEP their resource_effect so
    # that §3.6 (fixed instance enters chain via budget_debt) can fire.
    # We only zero out resource_effect when the instance is truly
    # state=dormant or has no LLM-supplied resource_effect at all.
    resource_effect = data.get("resource_effect")
    if not isinstance(resource_effect, dict):
        resource_effect = None
    elif state == "dormant":
        resource_effect = None
    # Note: state may legitimately be None (when fix_status is
    # fixed_at_step_<N>); in that case we keep resource_effect for
    # the fixed-but-costly path. exploration_suppressed will clear
    # it later if needed.
    if exploration_suppressed:
        resource_effect = None

    # Mechanical budget_debt: re-derive from wasted_step_count using
    # the §3.6 rule (wasted>5 => confirmed, 1..5 => possible, 0 => none).
    # We trust the LLM's wasted_step_count but ALWAYS override its
    # `budget_debt` enum so the downstream priority is reproducible.
    if isinstance(resource_effect, dict):
        try:
            wasted_count = int(resource_effect.get("wasted_step_count") or 0)
        except (TypeError, ValueError):
            wasted_count = 0
        if wasted_count < 0:
            wasted_count = 0
        resource_effect["wasted_step_count"] = wasted_count
        if wasted_count > 5:
            resource_effect["budget_debt"] = "confirmed"
        elif wasted_count >= 1:
            resource_effect["budget_debt"] = "possible"
        else:
            resource_effect["budget_debt"] = "none"

    fix_evidence = data.get("fix_evidence_quote") if fix_status != "active" else None
    state_evidence = data.get("state_evidence_quote") if state == "regular" else None

    # ---- qualified_origin_trigger_id / qualified_origin_taxonomy_tag ----
    qo_trigger_id = _coerce_int_or_none(data.get("qualified_origin_trigger_id"))
    qo_taxonomy_tag = str(data.get("qualified_origin_taxonomy_tag") or "").strip() or None

    return {
        "instance_id": instance_id,
        "origin_commitment_status": commit,
        "qualified_origin_step": qos,
        "qualified_origin_trigger_id": qo_trigger_id,
        "qualified_origin_taxonomy_tag": qo_taxonomy_tag,
        "origin_relocated": bool(origin_relocated),
        "exploration_suppressed": bool(exploration_suppressed),
        "fix_status": fix_status,
        "fix_evidence_quote": fix_evidence,
        "state": state,
        "state_evidence_quote": state_evidence,
        "semantic_status": semantic_status,
        "terminal_connection": terminal_connection,
        "chain_membership": bool(chain_membership),
        "resource_effect": resource_effect,
        "parse_failed": False,
        "state_confidence": "normal",
        "chain_explanation": str(data.get("chain_explanation", "") or "").strip(),
    }


def _default_result(instance: Dict[str, Any]) -> Dict[str, Any]:
    """Conservative parse-failure fallback per methodology §2.2 / §5.

    Triggered only after 3 perturbation retries have all failed. Under
    the v6 STRICT root-cause policy, an unparseable response means we
    cannot run the counterfactual test, so we default to the SAFE-OUT
    side: terminal_connection="none" / chain_membership=false. This
    matches the new Task 3 directive that "when in doubt, choose
    none". Phase 3 will treat the instance as non-chain (it can still
    appear in diagnostics, just not in chain_min_v2).
    """
    instance_id = int(instance.get("instance_id", 0))
    origin_step = _coerce_int_or_none(instance.get("origin_step")) or 0
    return {
        "instance_id": instance_id,
        "origin_commitment_status": "explicit_wrong_commitment",
        "qualified_origin_step": origin_step,
        "origin_relocated": False,
        "exploration_suppressed": False,
        "fix_status": "active",
        "fix_evidence_quote": None,
        "state": "regular",
        "state_evidence_quote": None,
        "semantic_status": "active",
        "terminal_connection": "none",
        "chain_membership": False,
        "resource_effect": None,
        "parse_failed": True,
        "state_confidence": "low",
        "qualified_origin_trigger_id": None,
        "qualified_origin_taxonomy_tag": None,
        "chain_explanation": (
            "LLM response could not be parsed after 3 perturbation "
            "retries; under the strict root-cause policy we default "
            "to terminal_connection=none / chain_membership=false "
            "(safe-out). Phase 3 should treat this as a non-chain "
            "candidate."
        ),
    }


# =====================================================================
# Phase 2 Annotator class
# =====================================================================


class Phase2Annotator:
    """Per-instance state annotator (v5 Phase 2)."""

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
        self.semaphore = asyncio.Semaphore(
            max(1, int(api_config.get("llm_concurrency", 10)))
        )
        self.post_trigger_window = int(
            api_config.get("post_trigger_th1_window", DEFAULT_POST_TRIGGER_TH1_WINDOW)
        )

    def close(self) -> None:
        self.model.close()

    async def _call_llm(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        usage_acc: Optional[Dict[str, int]] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        response_format = {"type": "json_object"} if self.judgement_force_json else None
        if max_tokens is None:
            max_tokens = int(self.config.get("max_tokens", 4096))
        else:
            max_tokens = int(max_tokens)
        if temperature is None:
            temperature = float(self.config.get("temperature", 0.0))
        if timeout is None:
            timeout = float(self.config.get(
                "llm_call_timeout", DEFAULT_LLM_CALL_TIMEOUT
            ))
        async with self.semaphore:
            result, usage = await asyncio.wait_for(
                asyncio.to_thread(
                    self.model.generate_chat,
                    messages,
                    max_tokens,
                    temperature,
                    response_format,
                    return_usage=True,
                ),
                timeout=timeout,
            )
        if usage and usage_acc is not None:
            usage_acc["input_tokens"] += usage.get("input_tokens", 0)
            usage_acc["reasoning_tokens"] += usage.get("reasoning_tokens", 0)
            usage_acc["output_tokens"] += usage.get("output_tokens", 0)
        return result or ""

    def _render_trajectory_for_instance(
        self,
        instance: Dict[str, Any],
        flat_steps: List[Dict[str, Any]],
        step_pool: Dict[int, Dict[str, str]],
    ) -> str:
        """Render the full trajectory with instance annotations inlined.

        Uses _render_trajectory_with_instance_inline to embed the
        instance's error_content / category / quotes directly after
        origin_step and last_trigger_step in the rendered output, so
        the LLM sees the Phase 1 evidence in context rather than as a
        separate JSON block.

        Per v5 spec: origin_step, last_trigger_step, and last_trigger_step + N
        subsequent steps are pinned at th1. Other steps follow compress_role /
        distance rules. The inline annotations are layered on top of
        this compression-aware rendering.
        """
        origin = int(instance.get("origin_step", 0))
        last_trigger = int(instance.get("last_trigger_step", origin))
        window_end = last_trigger + self.post_trigger_window

        # Extra th1 steps: origin, last_trigger, and N steps after last_trigger
        extra_th1 = [origin, last_trigger]
        for s in range(last_trigger + 1, window_end + 1):
            extra_th1.append(s)

        # Build a step_pool that has th1 content for the focal steps;
        # for all other steps we fall back to the raw flat_steps content
        # inside _render_trajectory_with_instance_inline.
        # We first obtain the compression-aware rendering per step by
        # constructing a per-step body map via render_history_for_focus,
        # then pass the step_pool (which already contains th1/th2/th3
        # keys) directly to the inline renderer so it can pick the
        # right tier for each step.
        #
        # Note: render_history_for_focus returns a single string; we
        # cannot easily split it back per step. Instead we pass the
        # step_pool directly — the inline renderer already knows how to
        # pick th1 > th2 > th3 > raw, which is the same priority order
        # used by render_history_for_focus for th1-pinned steps.
        return _render_trajectory_with_instance_inline(
            flat_steps=flat_steps,
            step_pool=step_pool,
            instance=instance,
        )

    async def annotate_instance(
        self,
        instance: Dict[str, Any],
        flat_steps: List[Dict[str, Any]],
        step_pool: Dict[int, Dict[str, str]],
        agent_framework_description: str,
        step_triggers: Optional[List[Dict[str, Any]]] = None,
        trajectory_outcome_block: Optional[str] = None,
        usage_acc: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """Annotate one instance with the v6 schema.

        Implements the methodology §2.2 / §5 retry policy: the LLM is
        called up to 3 times with deliberate PERTURBATIONS (temperature
        bump and schema-reminder strength change). Same-prompt /
        same-temperature triple-resampling is explicitly avoided
        because deterministic format failures would fail all three
        attempts. After the 3rd failure we fall through to
        ``_default_result``, which marks the instance parse_failed +
        active/regular + semantic_uncertain (low confidence).
        """
        rendered_trajectory = self._render_trajectory_for_instance(
            instance, flat_steps, step_pool
        )
        base_user_prompt = _build_user_prompt(
            agent_framework_description=agent_framework_description,
            instance=instance,
            rendered_trajectory=rendered_trajectory,
            step_triggers=step_triggers,
            trajectory_outcome_block=trajectory_outcome_block,
        )

        base_temp = float(self.config.get("temperature", 0.0))
        base_max_tokens = int(self.config.get("max_tokens", 4096))
        # 3-attempt retry policy with deterministic perturbation:
        #   * attempt 1: base temperature, base max_tokens
        #   * attempt 2: temperature + 0.3,  max_tokens * 1.5
        #   * attempt 3: temperature + 0.3,  base max_tokens
        # The schema reminder text is preserved unchanged from the
        # original 3-attempt plan so business behavior is identical;
        # only the (temperature, max_tokens) parameters are now
        # the primary perturbation knobs.
        attempts = [
            {
                "temperature": base_temp,
                "max_tokens": base_max_tokens,
                "reminder": "",
            },
            {
                "temperature": min(1.0, base_temp + 0.3),
                "max_tokens": int(base_max_tokens * 1.5),
                "reminder": (
                    "\n\nIMPORTANT: Return STRICT JSON only. Use ALL of "
                    "these keys: instance_id, origin_commitment_status, "
                    "qualified_origin_step, origin_relocated, "
                    "exploration_suppressed, fix_status, "
                    "fix_evidence_quote, state, state_evidence_quote, "
                    "semantic_status, terminal_connection, "
                    "chain_membership, resource_effect, "
                    "chain_explanation. No prose before or after the "
                    "JSON object."
                ),
            },
            {
                "temperature": min(1.0, base_temp + 0.3),
                "max_tokens": base_max_tokens,
                "reminder": (
                    "\n\nFINAL ATTEMPT. Output ONE JSON object and "
                    "NOTHING ELSE. The keys are exactly: "
                    "instance_id, origin_commitment_status, "
                    "qualified_origin_step, origin_relocated, "
                    "exploration_suppressed, fix_status, "
                    "fix_evidence_quote, state, state_evidence_quote, "
                    "semantic_status, terminal_connection, "
                    "chain_membership, resource_effect, "
                    "chain_explanation. Use null where appropriate."
                ),
            },
        ]

        for idx, plan in enumerate(attempts):
            user_prompt = base_user_prompt + plan["reminder"]
            try:
                response = await self._call_llm(
                    PHASE2_SYSTEM_PROMPT,
                    user_prompt,
                    temperature=plan["temperature"],
                    max_tokens=plan["max_tokens"],
                    usage_acc=usage_acc,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Phase2 LLM call TIMED OUT for instance %s on "
                    "attempt %d (%.1fs limit); will retry with "
                    "perturbation.",
                    instance.get("instance_id"), idx + 1,
                    float(self.config.get(
                        "llm_call_timeout", DEFAULT_LLM_CALL_TIMEOUT
                    )),
                )
                continue
            except Exception as exc:
                logger.warning(
                    "Phase2 LLM call failed for instance %s on attempt %d: %s",
                    instance.get("instance_id"), idx + 1, exc,
                )
                continue
            parsed = _parse_phase2_response(response, instance)
            if parsed is not None:
                if idx > 0:
                    parsed["_retry_count"] = idx
                return parsed
            logger.info(
                "Phase2 parse failed for instance %s on attempt %d; "
                "will perturb and retry (%d/%d).",
                instance.get("instance_id"), idx + 1, idx + 1, len(attempts),
            )

        # All 3 perturbation retries failed: conservative fallback.
        logger.warning(
            "Phase2 instance %s: all 3 retries failed; using "
            "parse_failed conservative fallback.",
            instance.get("instance_id"),
        )
        return _default_result(instance)

    async def annotate_trajectory(
        self,
        phase1_payload: Dict[str, Any],
        trajectory_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run Phase 2 on all instances (parallel)."""
        instances = phase1_payload.get("instances") or []
        flat_steps = trajectory_data["flat_steps"]
        step_pool = load_step_compressions_from_payload(phase1_payload)
        agent_fw = str(phase1_payload.get("agent_framework_description", "") or "")

        # Per-file token usage accumulator
        usage_acc: Dict[str, int] = {"input_tokens": 0, "reasoning_tokens": 0, "output_tokens": 0}

        if not instances:
            return {
                **_echo_payload_fields(phase1_payload),
                "instance_states": [],
                "token_usage": usage_acc,
            }

        step_triggers = phase1_payload.get("step_triggers") or []
        # Build a single trajectory-outcome block once and share it
        # across every instance call. This grounds Task 3 on the
        # OBSERVED terminal state (task_success / task_outcome / last
        # 2 steps) so the LLM stops imagining a successful trajectory.
        trajectory_outcome_block = _render_trajectory_outcome(
            task_success=phase1_payload.get("task_success"),
            task_outcome=phase1_payload.get("task_outcome"),
            flat_steps=flat_steps,
            step_pool=step_pool,
        )
        # Per-instance scheduling. The instance LLM calls are awaited
        # via asyncio.gather (so this trajectory's instances run in
        # parallel), but each individual call is bounded by
        # self.semaphore (the global llm_concurrency pool shared
        # across ALL trajectories) and by a wall-clock timeout inside
        # _call_llm. This means a single straggler instance only
        # holds ONE worker slot, freeing the rest of the pool to
        # process other trajectories' instances and eliminating the
        # "file_sem held hostage by one slow instance" pathology.
        tasks = [
            self.annotate_instance(
                inst, flat_steps, step_pool, agent_fw, step_triggers,
                trajectory_outcome_block=trajectory_outcome_block,
                usage_acc=usage_acc,
            )
            for inst in instances
        ]
        results: List[Dict[str, Any]] = await asyncio.gather(*tasks)

        # --- Post-processing: enforce v6 hard rules mechanically ---
        # The parser already enforces these per-instance, but we re-run
        # the checks here so that ANY downstream change to the parser
        # cannot accidentally relax the methodology guarantees. We also
        # keep span_exceeds_cap as an auxiliary signal for Phase 3.
        for inst, result in zip(instances, results):
            category = inst.get("category", "")
            origin_step = int(inst.get("origin_step", 0))
            last_trigger_step = int(inst.get("last_trigger_step", origin_step))
            span = last_trigger_step - origin_step

            # span_exceeds_cap is now folded into budget_debt as a
            # BACKSTOP for BOTH active/regular AND fixed-but-costly
            # instances: if the trigger span itself is > 5 steps and
            # the LLM did not fill a sufficient wasted_step_count, we
            # treat that span as wasted_step_count and set
            # budget_debt = confirmed. This is the sole way the
            # legacy span_exceeds_cap signal still influences chain
            # decisions, and the only way an LLM that forgot to label
            # a fixed-but-costly instance gets corrected.
            result["span_exceeds_cap"] = span > 5
            is_active_regular = (
                result.get("fix_status") == "active"
                and result.get("state") == "regular"
            )
            is_fixed = isinstance(
                result.get("fix_status"), str
            ) and result["fix_status"].startswith("fixed_at_step_")
            # For fixed instances we use (origin_step, fix_step] as the
            # waste window because steps after the fix are not wasted.
            effective_span = span
            if is_fixed:
                try:
                    fix_step_n = int(
                        result["fix_status"].replace("fixed_at_step_", "")
                    )
                    effective_span = max(0, min(
                        fix_step_n - origin_step,
                        last_trigger_step - origin_step,
                    ))
                except (TypeError, ValueError):
                    effective_span = span
            if (
                (is_active_regular or is_fixed)
                and effective_span > 5
                and not result.get("exploration_suppressed")
            ):
                re_obj = result.get("resource_effect") or {}
                if not isinstance(re_obj, dict):
                    re_obj = {}
                cur_wc = 0
                try:
                    cur_wc = int(re_obj.get("wasted_step_count") or 0)
                except (TypeError, ValueError):
                    cur_wc = 0
                if cur_wc < effective_span:
                    re_obj["wasted_step_count"] = effective_span
                    re_obj["budget_debt"] = (
                        "confirmed" if effective_span > 5
                        else ("possible" if effective_span >= 1 else "none")
                    )
                    re_obj.setdefault("wasted_steps", [])
                    re_obj.setdefault("last_referencing_step", last_trigger_step)
                    result["resource_effect"] = re_obj
                # If no terminal_connection was set or it was weak,
                # promote to budget_debt so chain_min_v2 can use this
                # instance. For fixed-but-costly instances this is
                # exactly the §3.6 chain entry path.
                if result.get("terminal_connection") in (None, "none", "semantic_uncertain"):
                    result["terminal_connection"] = "budget_debt"
                    result["chain_membership"] = True
                    if is_fixed:
                        logger.info(
                            "[post] fixed-but-costly backstop: instance %s "
                            "fix_status=%s span=%d -> terminal_connection="
                            "budget_debt, chain_membership=true",
                            result.get("instance_id"),
                            result.get("fix_status"),
                            effective_span,
                        )

            # Hard cat-1 no-dormant guarantee (defense in depth).
            # v6 strict policy: we still flip dormant -> regular
            # because cat-1 by definition violates a TASK clause at
            # T (observable terminal impact), but we DO NOT force
            # chain_membership=True any more. The LLM's strict
            # root-cause judgment on terminal_connection is
            # respected; if the LLM said "none" it means the cat-1
            # violation is dominated by some other cause, and we
            # leave it out of the chain.
            if (
                category == "cat-1"
                and result.get("fix_status") == "active"
                and result.get("state") == "dormant"
            ):
                logger.info(
                    "[post] cat-1 no-dormant override on instance %s",
                    result.get("instance_id"),
                )
                result["state"] = "regular"
                result["state_evidence_quote"] = None
                if result.get("qualified_origin_step") is None:
                    result["qualified_origin_step"] = origin_step
                    result["origin_commitment_status"] = "explicit_wrong_commitment"
                    result["exploration_suppressed"] = False

            # Hard env-fix re-fire invalidation (mechanical defense
            # for env-fix-1 / env-fix-2). If the LLM said fix_status =
            # fixed_at_step_<N> but the SAME env instance re-fired at
            # any step M > N (Phase 1 merged it into the same cluster
            # via domain / utm / gclid / error-code signature), the
            # supposed fix did not hold: roll back to active / regular
            # and let the span backstop below decide budget_debt.
            if category == "env" and is_fixed:
                try:
                    fix_step_n_env = int(
                        result["fix_status"].replace("fixed_at_step_", "")
                    )
                except (TypeError, ValueError):
                    fix_step_n_env = None
                # Collect all known re-fire step numbers for this
                # instance. Phase 1 emits last_trigger_step (largest
                # step among trigger_indices); some upstream paths
                # also surface a `trigger_steps` list. We treat any
                # step strictly greater than fix_step_n as a re-fire.
                refire_steps: List[int] = []
                lts = inst.get("last_trigger_step")
                if isinstance(lts, int):
                    refire_steps.append(lts)
                ts_list = inst.get("trigger_steps")
                if isinstance(ts_list, list):
                    for s in ts_list:
                        try:
                            refire_steps.append(int(s))
                        except (TypeError, ValueError):
                            continue
                if (
                    fix_step_n_env is not None
                    and any(s > fix_step_n_env for s in refire_steps)
                ):
                    logger.info(
                        "[post] env re-fix invalidation on instance %s: "
                        "fix_status=%s but last_trigger_step=%s -> "
                        "rolling back to active/regular",
                        result.get("instance_id"),
                        result.get("fix_status"),
                        max(refire_steps),
                    )
                    result["fix_status"] = "active"
                    result["state"] = "regular"
                    result["fix_evidence_quote"] = None
                    if result.get("semantic_status") == "fixed":
                        result["semantic_status"] = "active"
                    # v6 strict policy: do NOT force chain_membership
                    # on env re-fix invalidation. Rolling back to
                    # active/regular reopens the question; we let the
                    # span backstop below promote to budget_debt only
                    # when the trigger span actually exceeds the
                    # decisive-budget threshold (>5).
                    # Recompute is_fixed / is_active_regular flags so
                    # the downstream span backstop (which already ran
                    # above with the OLD fix status) gets re-applied
                    # locally for this instance.
                    is_fixed = False
                    is_active_regular = True
                    span_local = max(refire_steps) - origin_step
                    if span_local > 5 and not isinstance(
                        result.get("resource_effect"), dict
                    ):
                        result["resource_effect"] = {
                            "wasted_steps": [],
                            "wasted_step_count": span_local,
                            "budget_debt": "confirmed",
                            "last_referencing_step": max(refire_steps),
                        }
                        if result.get("terminal_connection") in (
                            None, "none", "semantic_uncertain",
                        ):
                            result["terminal_connection"] = "budget_debt"
                            result["chain_membership"] = True

            # Hard env no-pure_exploration guarantee (defense in
            # depth, symmetric to cat-1). An env instance is by
            # construction an already-observed environment violation.
            # Allowing pure_exploration on env silently drops genuine
            # critical errors (e.g. ad hijack, tool outage, stale API
            # response) whenever the AGENT's intent at origin_step
            # happened to be reasonable. The LLM should have used
            # fix_status=fixed_at_step_<N> for the env-recovers case
            # instead.
            if (
                category == "env"
                and result.get("origin_commitment_status") == "pure_exploration"
            ):
                logger.info(
                    "[post] env no-pure_exploration override on instance %s",
                    result.get("instance_id"),
                )
                result["origin_commitment_status"] = "explicit_wrong_commitment"
                result["qualified_origin_step"] = origin_step
                result["origin_relocated"] = False
                result["exploration_suppressed"] = False
                # Re-open the state decision. Previously pure_exploration
                # had forced active/dormant/none; we now let the
                # span-backstop below (or the existing fix/trigger
                # evidence) determine the real state. Default to
                # active/regular with a semantic_uncertain terminal
                # connection so Phase 3 can still see this instance;
                # the span backstop will upgrade to budget_debt if
                # the trigger span is long enough.
                if result.get("fix_status") == "active":
                    result["state"] = "regular"
                # v6 strict policy: do NOT default to
                # semantic_uncertain + chain_membership=True here.
                # The LLM's terminal_connection (including "none")
                # reflects the strict root-cause test. We only allow
                # the span backstop below to promote to budget_debt
                # if the wasted-step threshold is met.
                # Re-run the span backstop locally so the override
                # immediately benefits from budget_debt promotion
                # when (last_trigger_step - origin_step) > 5. We
                # also seed a minimal resource_effect from the
                # trigger span when the LLM emitted null.
                span_local = last_trigger_step - origin_step
                if span_local >= 1 and not isinstance(
                    result.get("resource_effect"), dict
                ):
                    result["resource_effect"] = {
                        "wasted_steps": [],
                        "wasted_step_count": span_local,
                        "budget_debt": (
                            "confirmed" if span_local > 5
                            else "possible"
                        ),
                        "last_referencing_step": last_trigger_step,
                    }
                    if span_local > 5 and result.get(
                        "terminal_connection"
                    ) in (None, "none", "semantic_uncertain"):
                        result["terminal_connection"] = "budget_debt"
                        result["chain_membership"] = True

            # Hard semantic-fixed exclusion guarantee (relaxed v6.3).
            # An instance whose semantic_status == "fixed" was repaired
            # before terminal. Such an instance leaves the chain ONLY
            # when BOTH of the following hold:
            #   (a) the LLM independently judged the instance has no
            #       terminal connection (terminal_connection == "none"),
            #       AND
            #   (b) the wasted_step_count is small (<= 5) so the repair
            #       did not consume material budget.
            # Compared to v6.0, we now KEEP sem=fixed instances when the
            # LLM still attached a connected terminal_connection
            # (semantic / irreversible / budget_debt) — those are
            # repaired-but-still-on-the-failure-path cases that the
            # strict v6.0 rule was incorrectly throwing out.
            if result.get("semantic_status") == "fixed" \
               and result.get("terminal_connection") == "none":
                re_obj = result.get("resource_effect") or {}
                wasted_for_check = 0
                if isinstance(re_obj, dict):
                    try:
                        wasted_for_check = int(re_obj.get("wasted_step_count") or 0)
                    except (TypeError, ValueError):
                        wasted_for_check = 0
                if wasted_for_check <= 5:
                    logger.info(
                        "[post] semantic-fixed exclusion (relaxed) on "
                        "instance %s: tc=none + wasted=%d -> "
                        "chain_membership=false",
                        result.get("instance_id"), wasted_for_check,
                    )
                    result["chain_membership"] = False

            # Multi-fire defensive promotion (v6.3).
            # If the LLM said terminal_connection == "none" but the
            # instance is still active and re-fires multiple times
            # after qualified_origin_step over a non-trivial span,
            # that pattern is direct observable evidence the wrong
            # commitment is carried to terminal — we promote
            # terminal_connection to "semantic" and chain_membership
            # to true. Gates kept narrow so this does not over-include
            # incidental triggers:
            #   * fix_status == "active" AND state == "regular"
            #   * not exploration_suppressed
            #   * not span_exceeds_cap (the span backstop already
            #     handled long-span cases above)
            #   * trigger_indices count >= 2
            #   * (last_trigger_step - qos) >= 3
            qos_for_promo = result.get("qualified_origin_step")
            if qos_for_promo is None:
                try:
                    qos_for_promo = int(result.get("qualified_origin_step") or origin_step)
                except (TypeError, ValueError):
                    qos_for_promo = origin_step
            try:
                qos_for_promo = int(qos_for_promo)
            except (TypeError, ValueError):
                qos_for_promo = origin_step
            if (
                result.get("terminal_connection") in (None, "none")
                and result.get("fix_status") == "active"
                and result.get("state") == "regular"
                and not result.get("exploration_suppressed")
                and not result.get("span_exceeds_cap")
            ):
                n_fires = len(inst.get("trigger_indices") or [])
                span_after_qos = last_trigger_step - qos_for_promo
                if n_fires >= 2 and span_after_qos >= 3:
                    logger.info(
                        "[post] multi-fire promotion on instance %s: "
                        "tc=none + fires=%d + span_after_qos=%d -> "
                        "tc=semantic, chain=True",
                        result.get("instance_id"), n_fires, span_after_qos,
                    )
                    result["terminal_connection"] = "semantic"
                    result["chain_membership"] = True

            # Hard pure_exploration suppression guarantee.
            if result.get("exploration_suppressed"):
                result["fix_status"] = "active"
                result["state"] = "dormant"
                result["qualified_origin_step"] = None
                result["terminal_connection"] = "none"
                result["chain_membership"] = False
                result["resource_effect"] = None

            # qualified_origin_step null ⇒ must not be in the chain.
            if result.get("qualified_origin_step") is None and not result.get(
                "exploration_suppressed"
            ):
                # Defensive: if the parser somehow produced a null QOS
                # without exploration_suppressed, treat it as suppressed.
                result["chain_membership"] = False
                if result.get("terminal_connection") != "none":
                    result["terminal_connection"] = "none"

        return {
            **_echo_payload_fields(phase1_payload),
            "instance_states": results,
            "phase2_schema_version": "v6",
            "token_usage": usage_acc,
        }

    async def process_file(
        self,
        phase1_file: str,
        trajectory_file: str,
        output_dir: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            with open(phase1_file, "r", encoding="utf-8") as fh:
                phase1_payload = json.load(fh)
        except Exception as exc:
            logger.error("Failed to load Phase 1 file %s: %s", phase1_file, exc)
            return None

        try:
            trajectory_data = load_unified_for_stage(trajectory_file)
        except Exception as exc:
            logger.error("Failed to load trajectory %s: %s", trajectory_file, exc)
            return None

        if phase1_payload.get("task_success"):
            payload = {
                **_echo_payload_fields(phase1_payload),
                "instance_states": [],
                "token_usage": {"input_tokens": 0, "reasoning_tokens": 0, "output_tokens": 0},
            }
        else:
            payload = await self.annotate_trajectory(phase1_payload, trajectory_data)

        stem = Path(phase1_file).stem.replace("_stage_c_phase1", "")
        out_fp = Path(output_dir) / f"{stem}_stage_c_phase2.json"
        with out_fp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        n_instances = len(payload.get("instance_states") or [])
        token_usage = payload.get("token_usage") or {}
        logger.info(
            "Phase 2 [%s]: %d instance states written (tokens: in=%d, reason=%d, out=%d)",
            stem, n_instances,
            token_usage.get("input_tokens", 0),
            token_usage.get("reasoning_tokens", 0),
            token_usage.get("output_tokens", 0),
        )
        return payload


def _echo_payload_fields(phase1_payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_id": phase1_payload.get("task_id"),
        "task_description": phase1_payload.get("task_description"),
        "task_success": phase1_payload.get("task_success"),
        "task_outcome": phase1_payload.get("task_outcome"),
        "reward": phase1_payload.get("reward"),
        "environment": phase1_payload.get("environment"),
        "total_steps": phase1_payload.get("total_steps"),
        "trajectory_source": phase1_payload.get("trajectory_source"),
        "agent_framework_description": phase1_payload.get("agent_framework_description"),
        "step_triggers": phase1_payload.get("step_triggers") or [],
        "step_compressions": phase1_payload.get("step_compressions") or {},
        "instances": phase1_payload.get("instances") or [],
        "trajectory_file_path": phase1_payload.get("trajectory_file_path") or "",
        "metadata": phase1_payload.get("metadata") or {},
    }


# =====================================================================
# Batch runner
# =====================================================================


def _is_valid_phase2_output(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return False
    return isinstance(data, dict) and "instance_states" in data


def _ensure_chat_completions_url(url: str) -> str:
    url = url.rstrip("/")
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url


_PHASE1_SUFFIX = "_stage_c_phase1.json"


def _collect_phase1_files(path: str) -> List[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        return [str(x) for x in sorted(p.rglob(f"*{_PHASE1_SUFFIX}"))]
    raise FileNotFoundError(f"Path not found: {path}")


def _index_trajectories_by_stem(trajectory_dir: str) -> Dict[str, str]:
    files = collect_trajectory_sources(trajectory_dir)
    return {output_stem_for_source(fp): fp for fp in files}


def _guess_trajectory_for_phase1(
    phase1_file: str,
    trajectory_index: Dict[str, str],
) -> Optional[str]:
    try:
        with open(phase1_file, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        ts = payload.get("trajectory_source")
        if isinstance(ts, str) and ts.strip():
            return ts
    except Exception:
        pass
    stem = Path(phase1_file).stem.replace("_stage_c_phase1", "")
    if stem in trajectory_index:
        return trajectory_index[stem]
    for key, val in trajectory_index.items():
        if key.startswith(stem) or stem.startswith(key):
            return val
    return None


async def run_batch(
    phase1_inputs: List[str],
    trajectory_dir: Optional[str],
    explicit_trajectory: Optional[str],
    output_dir: str,
    api_config: Dict[str, Any],
    concurrency: int = 4,
    resume: bool = False,
    overwrite: bool = False,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    api_config = dict(api_config)
    api_config["base_url"] = _ensure_chat_completions_url(api_config["base_url"])

    annotator = Phase2Annotator(api_config)
    trajectory_index = _index_trajectories_by_stem(trajectory_dir) if trajectory_dir else {}

    jobs: List[Tuple[str, str]] = []
    skipped_list: List[Dict[str, Any]] = []
    for p1 in phase1_inputs:
        if explicit_trajectory:
            traj = explicit_trajectory
        else:
            traj = _guess_trajectory_for_phase1(p1, trajectory_index)
        if not traj:
            skipped_list.append({"phase1_file": p1, "reason": "No matching trajectory found"})
            continue
        jobs.append((p1, traj))

    file_sem = asyncio.Semaphore(max(1, int(concurrency)))
    total = len(jobs)

    async def _process_one(idx: int, p1: str, traj: str) -> Tuple[str, str, str, Optional[str]]:
        stem = Path(p1).stem.replace("_stage_c_phase1", "")
        out_fp = Path(output_dir) / f"{stem}_stage_c_phase2.json"
        if overwrite:
            pass
        elif resume and _is_valid_phase2_output(out_fp):
            return ("skipped", p1, traj, f"valid cached: {out_fp.name}")
        async with file_sem:
            logger.info("Phase 2 (%d/%d): phase1=%s  traj=%s", idx, total, p1, traj)
            try:
                r = await annotator.process_file(p1, traj, output_dir)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                return ("failed", p1, traj, repr(exc))
        if r is None:
            return ("failed", p1, traj, "process_file returned None")
        return ("ok", p1, traj, None)

    ok = failed = 0
    failures: List[Dict[str, Any]] = []

    try:
        async_tasks = [_process_one(i, p1, traj) for i, (p1, traj) in enumerate(jobs, start=1)]
        results = await asyncio.gather(*async_tasks)
        for status, p1, traj, info in results:
            if status == "ok":
                ok += 1
            elif status == "failed":
                failed += 1
                failures.append({"phase1_file": p1, "trajectory_file": traj, "error": info or ""})
            else:
                skipped_list.append({"phase1_file": p1, "trajectory_file": traj, "reason": info or ""})
    finally:
        annotator.close()

    summary = {
        "output_dir": output_dir,
        "num_inputs": len(phase1_inputs),
        "num_jobs": len(jobs),
        "num_ok": ok,
        "num_failed": failed,
        "num_skipped": len(skipped_list),
        "concurrency": int(concurrency),
        "resume": resume,
        "overwrite": overwrite,
        "skipped": skipped_list[:20],
        "failures": failures[:20],
    }
    logger.info("Phase 2 batch done: %s", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage C Phase 2 (v5): Per-instance repair / state determination",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--phase1_file", help="Single Phase 1 result json")
    group.add_argument("--phase1_dir", help="Directory of Phase 1 results")

    parser.add_argument("--trajectory_file", help="Single original unified trajectory file")
    parser.add_argument("--trajectory_dir", help="Directory containing unified trajectory files")
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--api_key", default=os.getenv("API_KEY"))
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--judgement_force_json", type=lambda v: v.lower() not in ("0", "false", "no"), default=True)
    parser.add_argument("--llm_concurrency", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--llm_call_timeout",
        type=float,
        default=DEFAULT_LLM_CALL_TIMEOUT,
        help=(
            "Wall-clock timeout (seconds) for ONE LLM call within a "
            "Phase 2 instance attempt. Each instance is retried up "
            "to 3 times with deterministic perturbation; if all 3 "
            "attempts time out or fail to parse, the instance falls "
            "back to the conservative default result. Set higher for "
            "slow reasoning models or trajectories with very long "
            "prompts."
        ),
    )
    parser.add_argument("--post_trigger_th1_window", type=int, default=DEFAULT_POST_TRIGGER_TH1_WINDOW)
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
    if args.phase1_file and not args.trajectory_file:
        raise ValueError("--phase1_file requires --trajectory_file.")
    if args.phase1_dir and not args.trajectory_dir:
        raise ValueError("--phase1_dir requires --trajectory_dir.")

    api_config = {
        "api_key": args.api_key,
        "base_url": args.base_url,
        "model": args.model,
        "temperature": args.temperature,
        "cache_url": args.cache,
        "max_tokens": args.max_tokens,
        "judgement_force_json": args.judgement_force_json,
        "llm_concurrency": args.llm_concurrency,
        "llm_call_timeout": args.llm_call_timeout,
        "post_trigger_th1_window": args.post_trigger_th1_window,
        "model_profile": args.model_profile,
    }

    if args.phase1_file:
        inputs = [args.phase1_file]
    else:
        inputs = _collect_phase1_files(args.phase1_dir)

    summary = asyncio.run(
        run_batch(
            phase1_inputs=inputs,
            trajectory_dir=args.trajectory_dir,
            explicit_trajectory=args.trajectory_file,
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