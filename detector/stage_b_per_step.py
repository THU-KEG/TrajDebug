#!/usr/bin/env python3
"""Stage B (v5) — per-step trigger fire.

Per v5 spec §5.3 / §5.6.1, Stage B's only job is: for every judgable
step, decide which trigger tags (if any) fire. Each fired trigger
falls into one of four categories:

  cat-1 = the step conflicts with the TASK
  cat-2 = the step conflicts with VISIBLE CONTEXT
  cat-3 = the step is INTERNALLY INCONSISTENT
  env   = environment-side anomaly

Stage B does NOT maintain cross-step error state, does NOT decide
repair, does NOT cluster. Stage C handles those.

Inputs:
  * the Stage A pool file (``<stem>_stage_a.json``) for the three-tier
    compression pool;
  * the original unified trajectory file (already baked with
    ``judgable`` / ``compress_role`` per step by
    ``data_processing/bake_v5_fields``).

Output: ``<output_dir>/<stem>_stage_b.json`` carrying:
  * ``step_triggers`` — flat list of triggers (across all judgable
    steps) in v5 schema:
      step / category / taxonomy_tag / attribution /
      wrong_content_quote / reference_quote / confidence /
      confidence_reasoning;
  * the echoed pool so Stage C-Phase 1/2 can read everything from one
    file.

Stage B no longer reads ``stage_a`` diagnosis fields (Stage A no longer
emits them in v5). It does read the trajectory's
``agent_framework_description`` (from ``metadata.extra``) for the
prompt header.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from _stage_common import (
    dump_step_compressions,
    extract_last_json_object,
    load_step_compressions_from_payload,
    load_unified_for_stage,
    pool_line,
    render_history_for_focus,
    resolve_extra_params,
)
from utils.llm_compression import (
    DEFAULT_COMPRESSED_HISTORY_OVERALL_CAP_CHARS,
    DEFAULT_STEP_TH1_MAX_CHARS,
    DEFAULT_STEP_TH2_MAX_CHARS,
    DEFAULT_STEP_TH3_MAX_CHARS,
    strip_think_tags,
)
from utils.model import APIModel
from utils.trajectory_utils import (
    collect_trajectory_sources,
    output_stem_for_source,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 25 closed-vocabulary trigger tags (20 agent + 5 env). Stage B must
# never emit a tag outside this set.
VALID_TAXONOMY_TAGS = frozenset([
    # cat-1 / cat-2 / cat-3 (agent)
    "plan.BadDecomposition",
    "plan.WrongOrder",
    "plan.UnrealisticPlan",
    "plan.GoalDrift",
    "plan.OverExplore",
    "reason.WrongChoice",
    "reason.InvalidInference",
    "reason.MissingAssumptionCheck",
    "act.WrongTool",
    "act.ToolSchemaMismatch",
    "act.WrongActionSequence",
    "act.UnsafeOrForbiddenAction",
    "obs.MisreadOutput",
    "obs.IgnoreOutput",
    "obs.TimingIssue",
    "obs.GroundingFail",
    "verify.NoVerification",
    "verify.WrongVerification",
    "verify.PrematureTermination",
    "verify.InfiniteRetry",
    # env (environment-side anomaly)
    "env.AdOverlayHijack",
    "env.ContentFilterBlock",
    "env.ToolExtractorDegenerate",
    "env.RateLimitOrTransient",
    "env.EmptyOrRepeatedPayload",
])

ALLOWED_CATEGORIES = ("cat-1", "cat-2", "cat-3", "env")
ALLOWED_ATTRIBUTIONS = ("agent", "env")
ALLOWED_CONFIDENCE = ("high", "medium", "low")


@dataclass
class StageBTrigger:
    step: int
    category: str
    taxonomy_tag: str
    attribution: str
    wrong_content_quote: str
    reference_quote: str
    confidence: str
    confidence_reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": int(self.step),
            "category": self.category,
            "taxonomy_tag": self.taxonomy_tag,
            "attribution": self.attribution,
            "wrong_content_quote": self.wrong_content_quote,
            "reference_quote": self.reference_quote,
            "confidence": self.confidence,
            "confidence_reasoning": self.confidence_reasoning,
        }


# =====================================================================
# System prompt (v5 §5.6.1 — verbatim from the plan document)
# =====================================================================

STAGE_B_SYSTEM_PROMPT = r"""You are an error-trigger annotator for one step of an LLM-agent trajectory. For the CURRENT STEP, decide which trigger tags (if any) fire. Each fired trigger falls into one of four categories:

  cat-1 = the step conflicts with the TASK
  cat-2 = the step conflicts with VISIBLE CONTEXT (env / tool output / system feedback / prior environment-rendered fact)
  cat-3 = the step is INTERNALLY INCONSISTENT — its claim contradicts something the same message states verbatim, or contradicts the agent's own prior plan / reflection / memory
  env   = independent axis. Two sub-modes:
          (a) current-step env: the agent's tool call at THIS step is correct but the environment response is anomalous (see CHECKLIST (D));
          (b) upstream-env back-reference: the agent error at THIS step is rooted in an env anomaly at an EARLIER step (see STEP SCOPE).

NOTE on TASK scope: if HISTORY contains a role==system message, its binding constraints — statements using "must" / "shall" / "required" / "before X you must Y" / "only … if" / "never" / "forbidden", covering procedural rules, verification requirements, permission boundaries, ordering mandates — are part of the TASK for cat-1 purposes. Pure role descriptions like "You are an agent in …" are NOT binding constraints.

=== STEP SCOPE ===

Triggers normally have `step` == CURRENT STEP. The single exception is the upstream-env back-reference (env sub-mode (b)):

  Trigger condition: the CURRENT STEP makes an agent-side error (cat-1 / cat-2 / cat-3) whose load-bearing root cause is an env anomaly at an EARLIER step in HISTORY (the prior tool observation was empty / hijacked / rate-limited / degenerate-extractor / structurally insufficient, and the agent at THIS step fabricates or over-reaches because of it). When this fires you emit an ADDITIONAL trigger (the agent-side trigger at CURRENT STEP is still emitted as usual; both co-exist in the same `triggers` array).

  Constraints on the upstream-env trigger:
    (S1) `category="env"`, `attribution="env"`, tag from CHECKLIST (D), `step` = the earlier HISTORY step number.
    (S2) Quotes are taken from the EARLIER step, not CURRENT STEP (this overrides (C2) below):
           - `wrong_content_quote` = verbatim substring of that earlier step's environment payload as visible in HISTORY (the degenerate / empty / hijacked / rate-limited content itself);
           - `reference_quote`     = verbatim substring of TASK or of the prior agent / orchestrator instruction that established the sub-goal that earlier-step's tool call was supposed to serve.
    (S3) Causation is load-bearing: had the earlier observation been adequate, the current step would not have made this mistake. If the agent had alternative admissible actions (retry / different tool / scroll / open the result page) and simply failed to take them, do NOT fire — the error is purely agent-side at the current step.
    (S4) At most ONE upstream-env trigger per current-step analysis, pointing to the single earlier step that is the most direct anomalous source. Do not chain further back.
    (S5) Locality cap: the earlier step you point at must be NO FURTHER BACK than the immediately preceding agent turn. Concretely, walking backwards from CURRENT STEP through HISTORY, you may cross AT MOST ONE step with role=="assistant" (i.e. the closest prior assistant step) — the earlier step you cite must lie on or after that closest prior assistant step. If the anomalous source is older than that, do NOT fire the upstream-env trigger; the error is treated as purely agent-side at CURRENT STEP.

=== HARD RULES ===

(C1) Conflict, not suboptimality. The step must contradict a specific reference object I. "Could have done better" is NOT a trigger.

(C2) Two verbatim quotes per trigger:
       - wrong_content_quote : verbatim substring of the CURRENT STEP carrying the wrong content;
       - reference_quote     : verbatim substring of TASK / HISTORY / CURRENT STEP that is the I being conflicted.
     Paraphrase, summary, or composite quotes are not acceptable. If either quote cannot be cited verbatim, do NOT emit the trigger.
     EXCEPTION: the upstream-env back-reference (STEP SCOPE clause (S2)) sources its quotes from the earlier HISTORY step instead of CURRENT STEP. No other tag is allowed to deviate from (C2).
     NOTE (action-as-wrong-content): when the CURRENT STEP is (or contains) a tool call / action and the trigger is a conflict between the action itself and a rule or expected behaviour (e.g. act.UnsafeOrForbiddenAction against a system-policy clause, act.WrongActionSequence against a workflow constraint, act.ToolSchemaMismatch against a schema, act.WrongTool against a plan, reason.MissingAssumptionCheck where the load-bearing commitment IS the action), use the tool-call string verbatim (function name + arguments JSON / action text, exactly as it appears in the step) as `wrong_content_quote`, and the verbatim policy / workflow / schema / plan clause as `reference_quote`. No prose quote from the step is required.

(C3) Routing.
     - reference is environment / tool output / system state / prior environment-rendered fact -> CHECKLIST (B), cat-2.
     - reference is agent's own prior plan / reflection / memory / claim, or a same-message premise -> CHECKLIST (C), cat-3.
     act.WrongTool (b) is the historical exception: plan-action mismatch routes to CHECKLIST (B) / cat-2.

(C4) Admissible-action safe harbour. If the CURRENT STEP issues an action AND TASK / HISTORY provides an admissible / available / valid actions list, run this decision tree in order:

       Step 1. Is the action a verbatim member of the admissible list?
               NO  -> safe harbour does NOT apply.
               YES -> go to Step 2.

       Step 2. Does HISTORY contain the same action string (same name, same target, same params under JSON dict equality) followed by no-progress / error / no-relevant-state-change?
               YES -> safe harbour does NOT apply.
               NO  -> go to Step 3.

       Step 3. Is the action consistent with the plan stated in this step or in the immediately preceding agent step?
               NO  -> safe harbour does NOT apply.
               YES -> go to Step 4.

       Step 4. Is the plan/memory itself a product of information-ignorance? i.e. does the plan assert or imply a need (e.g. "I still need to obtain X") that is directly contradicted by visible information in HISTORY (e.g. X was already provided in a prior tool output), such that the plan would NOT have been formulated had that information been acknowledged?
               YES -> safe harbour does NOT apply for obs.IgnoreOutput (the plan is corrupted by the very omission this tag is designed to catch; consistency with a corrupted plan cannot shield the omission).
               NO  -> safe harbour APPLIES: do NOT fire any of obs.IgnoreOutput, verify.InfiniteRetry, act.WrongTool at this step, regardless of any omission you might otherwise spot in CHECKLIST (B). Other tags can still fire.

(C5) Commit and move on. Once you have identified a tag whose (C1)(C2)(C3) conditions are all satisfied and the safe-harbour check (C4) is resolved, commit to that tag immediately. Do NOT re-evaluate alternative tags for the same phenomenon more than once. If two tags both seem plausible for the same underlying error, pick the one that is closest to the root cause and stop deliberating.

=== CHECKLIST (A) — conflict with TASK (cat-1) ===

  [ ] The step's plan permanently drops a sub-task / sub-goal / deliverable the TASK explicitly requires, with no TODO / "later" / sequencing intent in the step itself, and HISTORY does not already cover it. Phased execution that still keeps deferred items in scope is NOT a trigger.
      -> plan.BadDecomposition

  [ ] The step's plan schedules a sub-task before another sub-task whose output it depends on, and TASK itself dictates this ordering.
      -> plan.WrongOrder

  [ ] The step depends on a tool / API / permission / data source that TASK or the environment does not provide.
      -> plan.UnrealisticPlan

  [ ] Double-anchor required. (i) HISTORY contains a verbatim line where the agent was previously on-constraint; (ii) THIS step contains a verbatim line that explicitly narrows / widens / replaces a TASK constraint. Both anchors must be in reference_quote. Gradual multi-step drift without an explicit single-step swerve is NOT a trigger here.
      -> plan.GoalDrift

  [ ] TASK explicitly requires verify / check / confirm of X, and THIS step delivers the final answer without performing that check. Implicit verification expectations are NOT a trigger here.
      -> verify.NoVerification

  [ ] The step declares completion / returns the final answer / stops the trajectory while a TASK-required deliverable is still unmet.
      ALSO FIRES, specifically, when the CURRENT STEP delivers a final answer / "TERMINATE" / final reply, AND any one of:
        (F1) Data-gap final answer. HISTORY contains a verbatim load-bearing data gap that the answer depends on — e.g. the answer requires "stops between A and B" but HISTORY only shows the segment between B and the line's far end (one side missing); the answer requires the value of a specific symbol but HISTORY's search results list other symbols' values without the queried one; the answer cites a source the agent never actually opened. The agent has not signalled "data is incomplete, I should look further". The wrong_content_quote is the final-answer assertion; the reference_quote is the verbatim HISTORY line showing the gap (the partial list / the search snippet that does NOT contain the queried key / etc.).
        (F2) Computed-without-input final answer. HISTORY does not contain (and the CURRENT STEP does not derive verbatim) one of the load-bearing inputs the final answer depends on, yet the step asserts a concrete answer. The wrong_content_quote is the asserted answer; the reference_quote is the TASK clause that names the missing input AND/OR the closest HISTORY line that confirms the input was never resolved.
      Implicit verification expectations alone are NOT a trigger here; the gap must be visible in HISTORY.
      -> verify.PrematureTermination

  [ ] The step performs an action that TASK or environment rules explicitly forbid (safety policy, permission boundary, forbidden endpoint, forbidden operation). This includes violations of binding constraints in any role==system message in HISTORY — e.g., performing a database-mutating action without the required prior verification / confirmation step mandated by the system policy.
      -> act.UnsafeOrForbiddenAction

=== CHECKLIST (B) — conflict with VISIBLE CONTEXT (cat-2) ===

  [ ] The step makes an explicit decision and cites supporting evidence, but the citation is distorted (wrong value, wrong field, mis-paraphrased), and the wrong reading drives the decision. Silent omission goes to obs.IgnoreOutput; correct citation but over-reaching conclusion goes to reason.InvalidInference.
      -> reason.WrongChoice

  [ ] The step cites visible evidence correctly, but generalises the conclusion beyond what the evidence supports (one data point -> whole class; one failure -> system broken).
      -> reason.InvalidInference

  [ ] The step relies on a prior tool call / observation / fact that does NOT actually appear in HISTORY (fabricated cross-message reference into a context that should exist).
      ALSO FIRES when: the step makes a decision or takes an action that depends on an unverified assumption, where:
        (i)   the agent has the means to verify (a tool / API is available and the required input is already known — e.g., user_id is known but get_user_details was never called; multiple orders exist but only one was checked);
        (ii)  the unverified assumption is load-bearing for the action taken (the action would differ if the assumption were false);
        (iii) TASK or HISTORY (including any role==system policy message) explicitly requires verification before this type of action, OR the assumption contradicts / is unsupported by available evidence.
      ALSO FIRES, specifically, in these patterns (each is a self-contained sufficient condition):
        (P1) Multi-candidate lock-in. HISTORY visibly enumerates N >= 2 candidate objects (e.g. "orders": ["#A","#B","#C","#D"]; multiple membership tiers; multiple pending bookings; multiple reservations on the user's profile). The CURRENT STEP commits to ONE of those candidates (queries it, modifies it, or recommends it) without any prior verification step that disambiguates which candidate matches the user's stated need (e.g. user listed items X/Y/Z but the agent only inspected one order). The wrong_content_quote is the lock-in tool call or commitment line; the reference_quote is the verbatim HISTORY line enumerating the candidates AND/OR the user's stated need that distinguishes between them.
        (P2) Skipped policy precondition. The HISTORY contains a binding system-policy / TASK clause requiring step X (verify user identity, list operation details, confirm with user, check eligibility) BEFORE step Y, AND the CURRENT STEP performs Y without first performing X. This applies even when X technically remains "doable later": the violation is that Y was issued without X. The wrong_content_quote is Y as it appears in this step (tool call, reply, action); the reference_quote is the verbatim policy / TASK clause requiring X first.
        (P3) Unconfirmed feasibility. The CURRENT STEP commits to or recommends a procedural option (e.g. "you can return only item Z from this order", "I'll upgrade only segment A"), AND HISTORY does NOT contain any verbatim policy / tool-output clause confirming the option is permitted for the specific object in question. "Plausible by analogy" is NOT enough; the option must be either (a) explicitly stated as available in TASK / HISTORY / system policy, or (b) demonstrably verified by a prior tool call. The wrong_content_quote is the option as committed in this step; the reference_quote is the user's request that the option is meant to satisfy AND/OR the policy clause that bounds the option space.
      IMPORTANT for (P1)/(P2)/(P3): these patterns are DESIGNED to apply to tool-call steps where the "wrong content" is the action ITSELF, not a prose claim. Per (C2) NOTE (action-as-wrong-content), use the tool-call string verbatim (function name + arguments JSON, exactly as it appears in the step) as `wrong_content_quote`. You do NOT need a prose quote from the step. Do NOT dismiss a (P1)/(P2)/(P3) match on the grounds that "the tool call doesn't contain an erroneous statement" — the conflict is between the action and the reference, not a textual contradiction inside the step.
      Exclusions: exploratory actions that will self-correct on the next tool response (e.g., trying one endpoint to see if it works) are NOT triggers unless the action is irreversible or TASK/policy mandates verification first. NOTE: this "exploratory" exclusion does NOT apply to (P1) when the candidate pool has N >= 2 disjoint items and querying one item yields information ONLY about that item (not about the other candidates) — because inspecting one order's contents tells you nothing about which of the other 3 orders contains the user's target items, so no "self-correction" signal is on its way. The "one endpoint to see if it works" exception is reserved for cases where the next tool response directly tells the agent whether to try a different endpoint (e.g., a 404 redirect, an error string naming the correct endpoint). Per-item detail lookups in a multi-item search space are NOT self-correcting and DO trigger (P1).
      -> reason.MissingAssumptionCheck

  [ ] (a) The chosen tool or data source is unsuited to the established sub-goal, AND one of: (i) visible evidence directly says so — HISTORY already tried this tool for this sub-goal and it failed; tool docs explicitly exclude this use; HISTORY contains an explicit "this tool can't do X" line; (ii) TASK or a prior plan / reflection step explicitly specifies a particular tool or data source (e.g. "use Google Finance", "search via Wikipedia"), but the current step uses a different one. "Better alternative existed" alone is NOT enough.
      Multi-agent note: when TASK names a specific tool / source / engine (e.g. "According to Google Finance ..."), an Orchestrator dispatch line that loosens it ("use X or another credible source") does NOT exempt the executing sub-agent from TASK. If the sub-agent's CURRENT STEP picks a source that is NOT the TASK-named one and the sub-agent has not first attempted the TASK-named source, this fires under (a)(ii). The wrong_content_quote is the sub-agent's tool / source choice; the reference_quote is the TASK's verbatim source clause.
      (b) Plan-action mismatch: this step (or the immediately prior agent step) commits to plan A, but the action issued is B which does not advance A.
      Subject to (C4).
      -> act.WrongTool

  [ ] A tool call is syntactically malformed against the visible schema or prior successful calls (missing fields, wrong types, enum out of range).
      -> act.ToolSchemaMismatch

  [ ] The action order violates a workflow established by HISTORY or the visible UI / env state machine (clicking submit before filling a required field that HISTORY shows still empty, git push before any commit, calling an API before authentication, navigating into a directory before listing it).
      -> act.WrongActionSequence

  [ ] The step misreads its own immediate tool / env output: treats an error as success, swaps fields, mis-identifies the returned entity, updates state from the wrong reading.
      -> obs.MisreadOutput

  [ ] Subject to (C4) -- BEFORE evaluating any sub-case below, you MUST first re-run the (C4) decision tree for this step. If safe harbour APPLIES, this whole tag MUST NOT fire, no matter how compelling the omission appears. Only after (C4) is unblocked do you proceed to one of the sub-cases:
        (i)   Information visible in current or prior tool / env output is silently ignored where it is clearly needed (no quote, no reference, no derived state). The step must also make a load-bearing wrong claim or wrong decision because of the ignored information; "could have cited more context" is NOT enough.
        (ii)  summary-omission: this step is itself a summary / memory-recap / status assessment / <memory> / <reflection> / <plan> block, AND either (ii-a) the summary makes a factual assertion that becomes incomplete or misleading because an individually identifiable fact / constraint / observation from HISTORY is dropped, OR (ii-b) the step omits a verifiable observation from HISTORY that is directly load-bearing for the current sub-goal (the omission causes or contributes to a wrong plan / action choice), even if the step makes no new factual assertion. The "merely quotes prior step(s) verbatim" exclusion no longer applies when the omission itself is load-bearing. reference_quote is the omitted HISTORY line.
        (iii) summary-oversimplification: this step is a summary that drops qualifying conditions / precision / magnitude from a HISTORY statement ("X holds under A, B, C" -> "X holds"; "approx. 30 mins" -> "30 mins") and uses the simplified version as a factual claim. reference_quote is the qualified HISTORY line. IMPORTANT: when a summary / memory block reduces a structured data payload (e.g. a table with multiple columns, a list with per-item details, a numeric record with magnitude) to a bare assertion like "the data was identified" or "the results were obtained" — discarding the actual content, column values, or magnitude — this IS summary-oversimplification. The wrong_content_quote should be the simplified assertion from the summary / memory block; the reference_quote should be the verbatim HISTORY data that was oversimplified (e.g. the table row with its full column values).
      LOAD-BEARING TEST (anti-noise). For ALL sub-cases above, the omission / mis-summary must be LOAD-BEARING for the step's plan or action choice. A trigger MUST NOT fire when the issue is purely cosmetic / metadata-level and does NOT alter what the agent decides to do next. Specifically these patterns are NOT triggers (do NOT fire obs.IgnoreOutput on them):
        (anti-i)   Memory recap mis-attributes a role label — e.g. the agent's memory block says 'Step 2: Action - "examine desk 1"' when step 2 was actually a user observation. If the agent's plan / next action is the same as it would have been with the correct attribution, this is cosmetic noise, NOT obs.IgnoreOutput.
        (anti-ii)  Memory recap drops a step but the dropped step's content is recapitulated elsewhere or is non-load-bearing for the current sub-goal.
        (anti-iii) The summary uses imprecise wording ("recently", "around") in a context where exact precision is not load-bearing.
      Before firing this tag you MUST be able to state, in confidence_reasoning, exactly WHICH downstream plan / action choice would have differed had the omission been corrected. If you cannot point to such a downstream consequence, do NOT fire.
      -> obs.IgnoreOutput

  [ ] The step treats a partial / pending / truncated response as final (visible "...", has_more=true, loading spinner still on, pagination markers still on).
      -> obs.TimingIssue

  [ ] The step binds intent to the wrong visible entity (clicks the wrong row, opens the wrong file, reads Q2 when sub-goal was Q3).
      -> obs.GroundingFail

  [ ] The step is performing verification but the criterion / target it checks does not match the sub-goal already established in HISTORY.
      -> verify.WrongVerification

  [ ] The step retries the same tool / query / action that already produced bad / empty / error in HISTORY, with no relevant env state change in between, and no strategy change. Subject to (C4).
      HARD prerequisite: HISTORY must contain AT LEAST ONE prior occurrence of the same action (same name, same target, same params under JSON-dict equality) followed by a visible bad-or-no-progress outcome (error string, empty payload, "Nothing happens.", state unchanged) in some earlier step. If this is the FIRST occurrence of this action in the trajectory, this tag MUST NOT fire — a first-time action is exploration, not retry. The wrong_content_quote is the CURRENT STEP's action; the reference_quote MUST cite both (a) the verbatim earlier-step action line AND (b) the verbatim earlier-step outcome line that demonstrates the bad outcome.
      "Bad outcome" is STRICT — it requires a literal error string / empty payload / "Nothing happens." / explicit state-unchanged signal. An observation that returned NEW information (item list, attribute values, entity details) but did NOT surface the specific object the agent was hoping to find is NOT a "bad outcome" for the purpose of this tag: the observation advanced state / informed the agent, even if it did not immediately satisfy the sub-goal. In other words, "the desired entity wasn't mentioned" does NOT make the prior observation a bad outcome — it makes it a partial / negative-evidence outcome, which is different. Do NOT fire verify.InfiniteRetry when the prior occurrence returned substantive new information.
      -> verify.InfiniteRetry

=== CHECKLIST (C) — internally inconsistent (cat-3) ===

  [ ] One of the following sub-cases. The conclusion contradicts something the SAME message states verbatim, OR contradicts a prior agent reflection / memory / claim (with reference object being agent-self, not environment data):
        (i)   the step lists N items and then asserts the count is M != N;
        (ii)  arithmetic / numeric / logical derivation whose inputs are cited verbatim in this message but the computed output is wrong (e.g. "A=5, B=3" -> "A+B=7");
        (iii) the step restates a premise P verbatim and then asserts a conclusion C that does not follow from P;
        (iv)  this step's reflection / memory / claim directly contradicts a prior agent reflection / memory / claim the agent itself produced (cross-message agent self-contradiction; reference is agent-self, not environment).
      -> reason.InvalidInference

  [ ] The step picks the wrong row / column / item from a list / table that is verbatim visible in the CURRENT MESSAGE itself (off-by-one, wrong row, wrong column).
      -> obs.MisreadOutput

  [ ] The step asserts a concrete factual claim drawn from the agent's parametric / internal knowledge, where ALL of:
        (a) the claim is concrete and falsifiable (not a hedge);
        (b) the claim is NOT supported by anything in TASK, HISTORY, or the CURRENT MESSAGE itself;
        (c) the claim is clearly wrong on widely-known ground truth ("the area of a circle is \u03c0r" -- wrong; "Paris is in Germany" -- wrong; "Python's sorted() returns None" -- wrong);
        (d) the wrong claim is load-bearing for this step's conclusion or next action.
      -> reason.MissingAssumptionCheck

=== CHECKLIST (D) — environment-side anomaly (env) ===

env triggers fire when the agent's tool call at this step is otherwise correct (right tool, right params, consistent with the established sub-goal), but the immediately following environment response shows the anomaly. Set attribution = "env" (not "agent"). env triggers do NOT count toward the agent failure chain.

  [ ] The search / page response is hijacked by an ad vignette / overlay, and the agent has no admissible action-space alternative.
      -> env.AdOverlayHijack

  [ ] The model API / platform content filter rejects a syntactically well-formed query.
      -> env.ContentFilterBlock

  [ ] The tool call itself is admissible (right tool, right params, consistent with the apparent sub-goal), but the returned observation is structurally insufficient because the tool is degenerate or blind on this target. Any one sub-case suffices:
        (i)   regardless of input, the tool returns a fixed literal / empty / repeated payload (e.g. SPA scrape returns only the page title);
        (ii)  the tool's output modality cannot, by construction, capture the information the sub-goal needs on this target -- e.g. a screenshot / OCR tool on a video watch page returns page chrome (title, sidebar, comments, recommendations) but cannot capture the actual video frames; an OCR of a search-results page returns navigation / ads / result-link headers but does NOT surface the answer text the sub-goal asks for (lyrics line, table cell value, embedded media content);
        (iii) the observation contains only structural framing of the page (header, navigation bar, sidebar, ad strip, cookie banner, "results for ..." heading) and contains NO content from the apparent target region, and this absence is a property of the tool x target pair rather than of the query string.
      When (ii) or (iii) fires, set `wrong_content_quote` to the verbatim portion of the observation that demonstrates the blindness (e.g. the chrome-only OCR block), and `reference_quote` to the sub-goal phrasing from TASK or the prior agent / orchestrator instruction that the tool failed to serve.
      -> env.ToolExtractorDegenerate

  [ ] The observation contains verbatim evidence of HTTP 429 / 5xx / timeout / connection reset / service unavailable / rate limited.
      -> env.RateLimitOrTransient

  [ ] The tool returns structurally empty / constant / payload identical to a prior call, with no clear structural reason (cache, jitter, etc.).
      -> env.EmptyOrRepeatedPayload

=== CONFIDENCE ===

For each trigger fill `confidence`:
  - high   : (C1)+(C2) are both crisp; wrong_content_quote and reference_quote are unambiguous and the conflict is direct.
  - medium : both quotes exist verbatim, but reference_quote needs light alignment (splitting a TASK clause, paraphrase-equivalence).
  - low    : inferring from a pattern; verbatim anchors are weak or partial.
Always fill `confidence_reasoning` with 1-2 sentences.

=== OUTPUT FORMAT ===

Emit a single JSON object:

{
  "triggers": [
    {
      "step": <int>,
      "category": "cat-1" | "cat-2" | "cat-3" | "env",
      "taxonomy_tag": "<one of the 25 tags listed in the checklists above>",
      "attribution": "agent" | "env",
      "wrong_content_quote": "<verbatim from CURRENT STEP, or — only for the upstream-env back-reference described in STEP SCOPE — verbatim from the earlier HISTORY step being pointed at>",
      "reference_quote": "<verbatim from TASK / HISTORY / CURRENT STEP>",
      "confidence": "high" | "medium" | "low",
      "confidence_reasoning": "<1-2 sentences>"
    }
  ]
}

`step` defaults to CURRENT STEP; the only case where it may point to an earlier step is the upstream-env back-reference (STEP SCOPE).

If no trigger fires, emit  {"triggers": []}.
"""


# =====================================================================
# User prompt builder
# =====================================================================


def _build_user_prompt(
    agent_framework_description: str,
    task_message: str,
    rendered_history: str,
    current_step: int,
    current_step_content: str,
) -> str:
    parts = []
    if agent_framework_description.strip():
        parts.append(f"=== AGENT FRAMEWORK ===\n{agent_framework_description.strip()}")
    parts.append(f"=== TASK ===\n{task_message.strip()}")
    parts.append(f"=== HISTORY ===\n{rendered_history}")
    parts.append(f"=== CURRENT STEP (msg {current_step}) ===\n{current_step_content}")
    return "\n\n".join(parts)


# =====================================================================
# Parser
# =====================================================================


def _parse_stage_b_response(
    response: str,
    step_num: int,
    min_allowed_back_step: Optional[int] = None,
) -> List[StageBTrigger]:
    """Parse the Stage B LLM response into validated triggers.

    ``min_allowed_back_step`` enforces v5 STEP SCOPE clause (S5): an
    upstream-env back-reference may only point to the closest prior
    assistant turn or later. When provided, any model-supplied
    back-reference ``step`` strictly less than this bound is coerced to
    ``step_num``. ``None`` disables the locality check (back-compat for
    callers like ``debug/debug_stage_b.py`` that lack the trajectory
    context).
    """
    response = strip_think_tags(response or "")
    if response.strip().startswith("```"):
        response = "\n".join(
            line for line in response.splitlines() if not line.strip().startswith("```")
        )

    data = extract_last_json_object(response, must_have_key="triggers")
    if data is None:
        data = extract_last_json_object(response)
    if not isinstance(data, dict):
        return []

    raw_triggers = data.get("triggers") or []
    if not isinstance(raw_triggers, list):
        return []

    triggers: List[StageBTrigger] = []
    for item in raw_triggers:
        if not isinstance(item, dict):
            continue
        tag = str(item.get("taxonomy_tag", "")).strip()
        if tag not in VALID_TAXONOMY_TAGS:
            logger.warning("Stage B step %s: dropped invalid tag=%r", step_num, tag)
            continue
        category = str(item.get("category", "")).strip()
        if category not in ALLOWED_CATEGORIES:
            if tag.startswith("env."):
                category = "env"
            elif tag.startswith("plan.") or tag.startswith("verify.") or tag.startswith("act.Unsafe"):
                category = "cat-1"
            else:
                category = "cat-2"
        attribution = str(item.get("attribution", "")).strip()
        if attribution not in ALLOWED_ATTRIBUTIONS:
            attribution = "env" if tag.startswith("env.") else "agent"
        wcq = str(item.get("wrong_content_quote", "") or "").strip()
        rq = str(item.get("reference_quote", "") or "").strip()
        if not wcq or not rq:
            logger.warning("Stage B step %s: dropped trigger with missing quote (tag=%s)", step_num, tag)
            continue
        confidence = str(item.get("confidence", "")).strip().lower()
        if confidence not in ALLOWED_CONFIDENCE:
            confidence = "low"
        confidence_reasoning = str(item.get("confidence_reasoning", "") or "").strip()

        # `step` defaults to the CURRENT STEP being analysed. Per the v5
        # "upstream-env back-reference" clause in STEP SCOPE, an env-category
        # trigger may legitimately point back to an EARLIER step in HISTORY
        # whose tool observation was anomalous and caused the current step's
        # error. We therefore accept a model-provided int `step` only when:
        #   * the value parses as int, AND
        #   * 0 <= value <= step_num (back-reference, never forward), AND
        #   * the trigger is on the env axis (category == "env" or tag is env.*).
        # All other cases fall back to step_num.
        emitted_step = step_num
        raw_step = item.get("step", None)
        if raw_step is not None:
            try:
                cand = int(raw_step)
            except (TypeError, ValueError):
                cand = None
            if (
                cand is not None
                and 0 <= cand <= step_num
                and (category == "env" or tag.startswith("env."))
            ):
                # v5 STEP SCOPE clause (S5): the back-reference must not
                # cross more than one prior assistant step. The lower
                # bound (closest prior assistant step number, or 0 if
                # none) is supplied by the caller via
                # ``min_allowed_back_step``.
                if (
                    min_allowed_back_step is not None
                    and cand < min_allowed_back_step
                    and cand != step_num
                ):
                    logger.warning(
                        "Stage B step %s: dropped back-reference step=%r (tag=%s) — older than closest prior assistant step %s; coercing to %s",
                        step_num, raw_step, tag, min_allowed_back_step, step_num,
                    )
                    emitted_step = step_num
                else:
                    emitted_step = cand
            elif cand is not None and cand != step_num:
                logger.warning(
                    "Stage B step %s: model emitted step=%r for non-env or invalid back-reference (tag=%s); coercing to %s",
                    step_num, raw_step, tag, step_num,
                )

        triggers.append(StageBTrigger(
            step=emitted_step,
            category=category,
            taxonomy_tag=tag,
            attribution=attribution,
            wrong_content_quote=wcq,
            reference_quote=rq,
            confidence=confidence,
            confidence_reasoning=confidence_reasoning,
        ))
    return triggers


# =====================================================================
# Detector class
# =====================================================================


class StageBDetector:
    """Per-step trigger fire detector (v5)."""

    def __init__(self, api_config: Dict[str, Any]):
        self.config = api_config
        cache_url = api_config.get("cache_url")
        if not cache_url:
            raise ValueError("Missing cache_url in api_config. Please provide --cache.")
        self.model = APIModel(
            cache_url,
            api_config["base_url"],
            api_config["model"],
            api_config.get("api_key", "EMPTY"),
            extra_params=resolve_extra_params(api_config.get("model_profile")),
        )
        self.history_overall_cap_chars = int(
            api_config.get("history_overall_cap_chars", DEFAULT_COMPRESSED_HISTORY_OVERALL_CAP_CHARS)
        )
        self.judgement_force_json = bool(api_config.get("judgement_force_json", True))
        self.semaphore = asyncio.Semaphore(
            max(1, int(api_config.get("llm_concurrency", 10)))
        )
        self.fallback_chars = {
            "th1": int(api_config.get("step_th1_max_chars", DEFAULT_STEP_TH1_MAX_CHARS)),
            "th2": int(api_config.get("step_th2_max_chars", DEFAULT_STEP_TH2_MAX_CHARS)),
            "th3": int(api_config.get("step_th3_max_chars", DEFAULT_STEP_TH3_MAX_CHARS)),
        }

    def close(self) -> None:
        self.model.close()

    async def _call_llm(self, system: str, user: str, usage_acc: Optional[Dict[str, int]] = None) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        response_format = {"type": "json_object"} if self.judgement_force_json else None
        max_tokens = int(self.config.get("max_tokens", 8192))
        temperature = float(self.config.get("temperature", 0.0))
        async with self.semaphore:
            result, usage = await asyncio.to_thread(
                self.model.generate_chat,
                messages,
                max_tokens,
                temperature,
                response_format,
                return_usage=True,
            )
        if usage and usage_acc is not None:
            usage_acc["input_tokens"] += usage.get("input_tokens", 0)
            usage_acc["reasoning_tokens"] += usage.get("reasoning_tokens", 0)
            usage_acc["output_tokens"] += usage.get("output_tokens", 0)
        return result or ""

    async def analyze_step(
        self,
        step_data: Dict[str, Any],
        flat_steps: List[Dict[str, Any]],
        step_pool: Dict[int, Dict[str, str]],
        task_description: str,
        agent_framework_description: str,
        usage_acc: Optional[Dict[str, int]] = None,
    ) -> List[StageBTrigger]:
        """Analyze one step. Returns 0-N triggers."""
        step_num = int(step_data["step"])

        # Render current step at th1
        pool_entry = step_pool.get(step_num) or {}
        current_content = (
            pool_entry.get("th1")
            or pool_entry.get("th2")
            or pool_entry.get("th3")
            or str(step_data.get("content", "") or "")
        )

        # Render history using the v5 tier router
        warn_state: List[bool] = [False]
        rendered_history = render_history_for_focus(
            flat_steps=flat_steps,
            focus_step=step_num,
            step_pool=step_pool,
            warn_state=warn_state,
            fallback_chars=self.fallback_chars,
            th1_max_distance=2,
            th2_max_distance=5,
            include_focus=False,
            history_only_before=True,
        )

        user_prompt = _build_user_prompt(
            agent_framework_description=agent_framework_description,
            task_message=task_description,
            rendered_history=rendered_history,
            current_step=step_num,
            current_step_content=current_content,
        )

        # Compute the lower bound for an upstream-env back-reference
        # (v5 STEP SCOPE clause (S5)): walking backward from the current
        # step, the trigger may point no further back than the closest
        # prior assistant step. If there is no prior assistant step,
        # fall back to 0 (i.e. no extra restriction beyond the existing
        # 0 <= step <= step_num bound).
        min_allowed_back_step = 0
        for prior in flat_steps:
            try:
                prior_sid = int(prior.get("step"))
            except (TypeError, ValueError):
                continue
            if prior_sid >= step_num:
                continue
            prior_role = str(
                prior.get("message_role") or prior.get("role") or ""
            ).lower()
            if prior_role == "assistant" and prior_sid > min_allowed_back_step:
                min_allowed_back_step = prior_sid

        response = await self._call_llm(STAGE_B_SYSTEM_PROMPT, user_prompt, usage_acc=usage_acc)
        triggers = _parse_stage_b_response(
            response,
            step_num,
            min_allowed_back_step=min_allowed_back_step,
        )

        # Post-filter: any trigger whose step is not judgable is invalid —
        # drop it regardless of whether it is a back-reference or current step.
        step_judgable: Dict[int, bool] = {
            int(s["step"]): bool(s.get("judgable", True))
            for s in flat_steps
        }
        filtered: List[StageBTrigger] = []
        for t in triggers:
            if not step_judgable.get(t.step, True):
                logger.warning(
                    "Stage B step %s: dropped trigger step=%s (tag=%s) "
                    "— that step is not judgable",
                    step_num, t.step, t.taxonomy_tag,
                )
                continue
            filtered.append(t)
        return filtered

    async def analyze_trajectory(
        self,
        stage_a_payload: Dict[str, Any],
        trajectory_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run Stage B on all judgable steps (parallel)."""
        task_description = trajectory_data["task_description"]
        flat_steps = trajectory_data["flat_steps"]
        agent_framework_description = trajectory_data["agent_framework_description"]
        step_pool = load_step_compressions_from_payload(stage_a_payload)

        # Per-file token usage accumulator
        usage_acc: Dict[str, int] = {"input_tokens": 0, "reasoning_tokens": 0, "output_tokens": 0}

        tasks = []
        for step in flat_steps:
            if not step.get("judgable", True):
                continue
            tasks.append(self.analyze_step(
                step_data=step,
                flat_steps=flat_steps,
                step_pool=step_pool,
                task_description=task_description,
                agent_framework_description=agent_framework_description,
                usage_acc=usage_acc,
            ))

        all_trigger_lists: List[List[StageBTrigger]] = await asyncio.gather(*tasks)
        all_triggers: List[Dict[str, Any]] = []
        for trig_list in all_trigger_lists:
            for t in trig_list:
                all_triggers.append(t.to_dict())

        return {
            "task_id": trajectory_data["task_id"],
            "task_description": task_description,
            "task_success": trajectory_data["task_success"],
            "task_outcome": trajectory_data["task_outcome"],
            "reward": trajectory_data["reward"],
            "environment": trajectory_data["environment"],
            "total_steps": trajectory_data["total_steps"],
            "trajectory_source": trajectory_data["trajectory_source"],
            "agent_framework_description": agent_framework_description,
            "step_triggers": all_triggers,
            "step_compressions": dump_step_compressions(step_pool),
            "trajectory_file_path": trajectory_data.get("file_path") or trajectory_data.get("source_file") or "",
            "metadata": {
                "dataset": trajectory_data.get("metadata", {}).get("dataset"),
                "annotation": trajectory_data.get("metadata", {}).get("annotation"),
                "extra": trajectory_data.get("metadata", {}).get("extra"),
            },
            "token_usage": usage_acc,
        }

    async def process_file(
        self,
        stage_a_file: str,
        trajectory_file: str,
        output_dir: str,
    ) -> Optional[Dict[str, Any]]:
        try:
            with open(stage_a_file, "r", encoding="utf-8") as fh:
                stage_a_payload = json.load(fh)
        except Exception as exc:
            logger.error("Failed to load Stage A file %s: %s", stage_a_file, exc)
            return None

        try:
            trajectory_data = load_unified_for_stage(trajectory_file)
        except Exception as exc:
            logger.error("Failed to load trajectory %s: %s", trajectory_file, exc)
            return None

        if trajectory_data["task_success"]:
            payload: Dict[str, Any] = {
                "task_id": trajectory_data["task_id"],
                "task_description": trajectory_data["task_description"],
                "task_success": True,
                "task_outcome": "success",
                "reward": trajectory_data["reward"],
                "environment": trajectory_data["environment"],
                "total_steps": trajectory_data["total_steps"],
                "trajectory_source": trajectory_data["trajectory_source"],
                "agent_framework_description": trajectory_data["agent_framework_description"],
                "step_triggers": [],
                "step_compressions": stage_a_payload.get("step_compressions") or {},
                "token_usage": {"input_tokens": 0, "reasoning_tokens": 0, "output_tokens": 0},
            }
        else:
            payload = await self.analyze_trajectory(stage_a_payload, trajectory_data)

        out_stem = output_stem_for_source(trajectory_file)
        out_fp = Path(output_dir) / f"{out_stem}_stage_b.json"
        with out_fp.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        n_triggers = len(payload.get("step_triggers") or [])
        token_usage = payload.get("token_usage") or {}
        logger.info(
            "Stage B [%s]: %d triggers across %d steps (tokens: in=%d, reason=%d, out=%d)",
            out_stem, n_triggers, payload["total_steps"],
            token_usage.get("input_tokens", 0),
            token_usage.get("reasoning_tokens", 0),
            token_usage.get("output_tokens", 0),
        )
        return payload


# =====================================================================
# Batch runner
# =====================================================================


def _is_valid_stage_b_output(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return False
    if not isinstance(data, dict):
        return False
    if "step_triggers" not in data:
        return False
    if not isinstance(data["step_triggers"], list):
        return False
    return True


def _ensure_chat_completions_url(url: str) -> str:
    url = url.rstrip("/")
    if url.endswith("/v1"):
        return url + "/chat/completions"
    return url


def _index_trajectories_by_stem(trajectory_dir: str) -> Dict[str, str]:
    files = collect_trajectory_sources(trajectory_dir)
    return {output_stem_for_source(fp): fp for fp in files}


_STAGE_A_SUFFIX = "_stage_a.json"


def _strip_stage_a_suffix(stem: str) -> str:
    if stem.endswith("_stage_a"):
        return stem[: -len("_stage_a")]
    return stem


def _guess_trajectory_for_stage_a(
    stage_a_file: str,
    trajectory_index: Dict[str, str],
) -> Optional[str]:
    try:
        with open(stage_a_file, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        ts = payload.get("trajectory_source")
        if isinstance(ts, str) and ts.strip():
            return ts
    except Exception:
        pass
    stem = _strip_stage_a_suffix(Path(stage_a_file).stem)
    if stem in trajectory_index:
        return trajectory_index[stem]
    for key, val in trajectory_index.items():
        if key.startswith(stem) or stem.startswith(key):
            return val
    return None


async def run_batch(
    stage_a_inputs: List[str],
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

    detector = StageBDetector(api_config)
    trajectory_index = _index_trajectories_by_stem(trajectory_dir) if trajectory_dir else {}

    jobs: List[Tuple[str, str]] = []
    skipped_list: List[Dict[str, Any]] = []
    for sa in stage_a_inputs:
        if explicit_trajectory:
            traj = explicit_trajectory
        else:
            traj = _guess_trajectory_for_stage_a(sa, trajectory_index)
        if not traj:
            skipped_list.append({"stage_a_file": sa, "reason": "No matching trajectory found"})
            continue
        jobs.append((sa, traj))

    file_sem = asyncio.Semaphore(max(1, int(concurrency)))
    total = len(jobs)

    async def _process_one(idx: int, sa: str, traj: str) -> Tuple[str, str, str, Optional[str]]:
        out_fp = Path(output_dir) / f"{output_stem_for_source(traj)}_stage_b.json"
        if overwrite:
            pass
        elif resume and _is_valid_stage_b_output(out_fp):
            return ("skipped", sa, traj, f"valid cached: {out_fp.name}")
        async with file_sem:
            logger.info("Processing (%d/%d): stage_a=%s  traj=%s", idx, total, sa, traj)
            try:
                r = await detector.process_file(sa, traj, output_dir)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                return ("failed", sa, traj, repr(exc))
        if r is None:
            return ("failed", sa, traj, "process_file returned None")
        return ("ok", sa, traj, None)

    ok = failed = 0
    failures: List[Dict[str, Any]] = []
    try:
        async_tasks = [_process_one(i, sa, traj) for i, (sa, traj) in enumerate(jobs, start=1)]
        results = await asyncio.gather(*async_tasks)
        for status, sa, traj, info in results:
            if status == "ok":
                ok += 1
            elif status == "failed":
                failed += 1
                failures.append({"stage_a_file": sa, "trajectory_file": traj, "error": info or ""})
            else:
                skipped_list.append({"stage_a_file": sa, "trajectory_file": traj, "reason": info or ""})
    finally:
        detector.close()

    summary = {
        "output_dir": output_dir,
        "num_stage_a_inputs": len(stage_a_inputs),
        "num_jobs": len(jobs),
        "num_ok": ok,
        "num_failed": failed,
        "num_skipped": len(skipped_list),
        "file_concurrency": int(concurrency),
        "llm_concurrency": int(api_config.get("llm_concurrency", 10)),
        "resume": resume,
        "overwrite": overwrite,
        "skipped": skipped_list[:20],
        "failures": failures[:20],
    }
    logger.info("Stage B batch done: %s", summary)
    return summary


def _collect_stage_a_files(path: str) -> List[str]:
    p = Path(path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        return [str(x) for x in sorted(p.rglob(f"*{_STAGE_A_SUFFIX}"))]
    raise FileNotFoundError(f"Path not found: {path}")


_BOOL_FALSE_STRINGS = {"0", "false", "no", "off", "n", "f"}


def _parse_bool_flag(value: Any) -> bool:
    return str(value).strip().lower() not in _BOOL_FALSE_STRINGS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage B (v5): Per-step trigger fire detection",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage_a_file", help="Single Stage A result json (*_stage_a.json)")
    group.add_argument("--stage_a_dir", help="Directory containing Stage A result json files")

    parser.add_argument("--trajectory_file", help="Single original unified trajectory file")
    parser.add_argument("--trajectory_dir", help="Directory containing unified trajectory files")
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--api_key", default=os.getenv("API_KEY"))
    parser.add_argument("--base_url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--cache", required=True)

    parser.add_argument("--history_overall_cap_chars", type=int, default=DEFAULT_COMPRESSED_HISTORY_OVERALL_CAP_CHARS)
    parser.add_argument("--judgement_force_json", type=_parse_bool_flag, default=True)
    parser.add_argument("--max_tokens", type=int, default=8192)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--llm_concurrency", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--model_profile",
        default=None,
        help=(
            "Optional model profile name from utils/model_profiles.py. "
            "When set, the profile's `params` (excluding temperature / "
            "max_tokens which stay owned by CLI) are forwarded to the "
            "OpenAI-compatible API as extra request parameters."
        ),
    )

    args = parser.parse_args()
    if not args.api_key:
        raise ValueError("Missing --api_key and env API_KEY is not set.")
    if args.stage_a_file and not args.trajectory_file:
        raise ValueError("--stage_a_file requires --trajectory_file.")
    if args.stage_a_dir and not args.trajectory_dir:
        raise ValueError("--stage_a_dir requires --trajectory_dir.")

    api_config = {
        "api_key": args.api_key,
        "base_url": args.base_url,
        "model": args.model,
        "temperature": args.temperature,
        "max_retries": args.max_retries,
        "timeout": args.timeout,
        "cache_url": args.cache,
        "max_tokens": args.max_tokens,
        "history_overall_cap_chars": args.history_overall_cap_chars,
        "judgement_force_json": args.judgement_force_json,
        "llm_concurrency": args.llm_concurrency,
        "model_profile": args.model_profile,
    }

    if args.stage_a_file:
        stage_a_inputs = [args.stage_a_file]
    else:
        stage_a_inputs = _collect_stage_a_files(args.stage_a_dir)

    summary = asyncio.run(
        run_batch(
            stage_a_inputs=stage_a_inputs,
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
