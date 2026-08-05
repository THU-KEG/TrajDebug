#!/usr/bin/env python3
"""
Error Definitions Loader - Taxonomy-aligned (English only)
"""

from typing import Dict, Any, List


class ErrorDefinitionsLoader:
    """Loads and formats error definitions for prompts."""

    def __init__(self):
        self.definitions = self._load_definitions()

    def _load_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Load error definitions aligned to taxonomy.md (English only)."""
        return {
            "plan": {
                "guidance": "PLAN focuses on the agent's multi-step strategy and task decomposition. Do not judge low-level tool syntax here; if the plan is sound but execution fails due to a malformed tool call, that belongs to ACT.",
                "errors": {
                    "BadDecomposition": {
                        "definition": "The agent decomposes the task into sub-tasks incorrectly, such that key sub-tasks are missing, redundant, or wrongly specified. This includes omitting a required step (e.g., not checking a necessary constraint) or defining a subgoal that does not contribute to the final objective.",
                        "example": "Task: 'Book a 30-minute meeting with Alex next week; avoid Tue; use the calendar tool; propose 3 options; include a Zoom link.' The agent's plan is: '(1) Ask Alex for availability, (2) schedule meeting.' It omits key sub-tasks: checking the user's calendar conflicts, enforcing 'avoid Tue,' generating 3 candidate slots, and adding a Zoom link.",
                    },
                    "WrongOrder": {
                        "definition": "The agent chooses an incorrect order of steps such that later actions depend on earlier steps that were not completed, or conclusions are drawn before collecting necessary evidence. The workflow ordering prevents successful execution or produces invalid results.",
                        "example": "Task: 'Book a 30-minute meeting with Alex next week; avoid Tue; use the calendar tool; propose 3 options; include a Zoom link.' The agent immediately creates the calendar event before checking conflicts and before generating 3 options. After creation fails due to conflict/Tue restriction, it backtracks repeatedly. The ordering prevents smooth execution.",
                    },
                    "UnrealisticPlan": {
                        "definition": "The agent forms a plan that relies on non-existent tools, unavailable APIs, inaccessible websites, missing credentials/permissions, or information sources that the environment cannot provide. The plan is infeasible given the tools and constraints.",
                        "example": "Task: 'Book a 30-minute meeting with Alex next week; avoid Tue; use the calendar tool; propose 3 options; include a Zoom link.' The agent plans to 'query Alex's private calendar directly via API' even though the environment only provides the user's calendar tool and has no access to Alex's calendar. The plan depends on an unavailable capability.",
                    },
                    "GoalDrift": {
                        "definition": "The agent gradually shifts away from the original task objective or constraints, pursuing a different question, relaxing requirements (e.g., ignoring region, season count), or focusing on irrelevant details. The plan no longer targets the correct deliverable.",
                        "example": "Task: 'Book a 30-minute meeting with Alex next week; avoid Tue; use the calendar tool; propose 3 options; include a Zoom link.' Instead of proposing 3 time options next week (excluding Tue) and including a Zoom link, the agent starts negotiating meeting agenda details and sends a long email draft without producing time options or a scheduling action.",
                    },
                    "OverExplore": {
                        "definition": "The agent performs unnecessary exploration, excessive searching, or redundant branching that does not materially increase the probability of success, causing wasted steps, cost, or context. The exploration is disproportionate to the task needs.",
                        "example": "Task: 'Book a 30-minute meeting with Alex next week; avoid Tue; use the calendar tool; propose 3 options; include a Zoom link.' After already finding 5 valid free slots and preparing 3 options, the agent continues scanning multiple weeks and exploring 'best meeting practices,' consuming steps without improving the scheduling outcome.",
                    },
                },
            },
            "reason": {
                "guidance": "REASON focuses on single-step reasoning and decision making. REASON errors are about how the agent converts evidence into a decision in this step, not about whether the agent used the right tool (ACT) or misread the output (OBS). If the evidence is read correctly but the conclusion is still wrong, that is REASON.",
                "errors": {
                    "WrongChoice": {
                        "definition": "The agent's conclusion is driven by evidence that IS visible in the step (task statement, policy, prior tool output, or prior step still shown in the history) but the agent IGNORES that evidence or QUOTES IT INCORRECTLY. Typical forms: violating a task/policy constraint that is stated verbatim; restating a prior numeric or categorical field with the wrong value; summarizing earlier steps in a way that drops or distorts a still-visible key detail; attributing a failure to an unrelated cause while an explicit error-message cue is right there in the output. The defining feature is that the supporting evidence is present and legible, yet the decision is built on a distorted or discarded version of it.",
                        "example": "Example 1. Task: 'Choose the cheapest shipping option that arrives by Friday. A: $12 Thu; B: $8 Sat; C: $10 Fri.' The agent picks option B because it is cheapest, ignoring the Friday constraint that is stated verbatim in the task. Example 2. The previous tool output shows total=17, but in the current step the agent writes 'previous total was 7' and plans around 7; the original value is still visible in the prior step, yet the agent mis-quotes it.",
                    },
                    "InvalidInference": {
                        "definition": "The evidence is visible AND the agent cites it accurately in this step, yet the conclusion over-reaches what that evidence can support. Typical forms: inferring that an entire tool, site, or strategy is broken from a single failed call; concluding a general rule from one data point; jumping from 'A was observed' to 'therefore B must hold' without a supporting link. The agent reads the evidence correctly; the failure is in the logical leap from evidence to conclusion, not in what the evidence says.",
                        "example": "Example 1. Task: 'Choose the cheapest shipping option that arrives by Friday. A: $12 Thu; B: $8 Sat; C: $10 Fri.' The agent correctly lists all three prices and deadlines, then concludes 'A is the cheapest Friday-arriving option'. The evidence is read correctly but the inference over-reaches, since C also satisfies Friday and is cheaper than A. Example 2. After a single query returns no results, the agent concludes the entire search service is down, even though one empty result does not support that.",
                    },
                    "MissingAssumptionCheck": {
                        "definition": "The agent relies on a claimed fact that has NO supporting evidence in the current step or any visible prior step. Typical forms: stating that a prior step observed X when no such prior step exists in the available context; recalling a tool call or tool result that was never produced; reflecting on a failure that never happened; proceeding from memory or guess for a value the agent would have needed to re-retrieve but didn't. The defining feature is that the supporting evidence for the claim does not exist in the available context — the agent has not checked, retrieved, or verified it.",
                        "example": "Example 1. Task: 'Choose the cheapest shipping option that arrives by Friday.' The agent writes 'Saturday delivery counts as arriving by Friday' and proceeds; it never checks the wording of the deadline — an unchecked assumption. Example 2. The agent's thought includes 'Based on my earlier call to the search tool, no matching items were found, so I will switch to browsing'; but the trajectory shown so far contains no prior call to any search tool, so the recalled event does not exist in the available context, yet the agent plans as if it did.",
                    },
                },
            },
            "act": {
                "guidance": "ACT focuses on actions and tool usage. ACT errors are about doing the right kind of operation in the right form. Do not penalize an action as ACT if it is blocked only due to external service failures unrelated to the agent's behavior. If the action is disallowed and explicitly rejected or blocked by environment rules or safety policy, that is UnsafeOrForbiddenAction.",
                "errors": {
                    "WrongTool": {
                        "definition": "The agent uses an inappropriate tool or action type for the task context. The chosen tool cannot produce the needed information or is mismatched to the environment requirements.",
                        "example": "Task in a web automation environment: 'On the site, search for 'wireless mouse', filter price <= $25, open the first result, add to cart.' Instead of using the browser/search tool to interact with the site, the agent writes a natural-language guess of products and prices and outputs a final recommendation without performing any site actions.",
                    },
                    "ToolSchemaMismatch": {
                        "definition": "The agent issues a tool call with invalid formatting or parameters that do not match the expected schema (missing required fields, wrong types, malformed JSON). The tool call fails or cannot be parsed.",
                        "example": "Task in a web automation environment: 'On the site, search for 'wireless mouse', filter price <= $25, open the first result, add to cart.' The environment expects a tool call like {'action':'click','selector':'#add-to-cart'} but the agent outputs malformed JSON or missing fields (e.g., {'selector':'#add-to-cart'}), causing the tool call to be rejected.",
                    },
                    "WrongActionSequence": {
                        "definition": "The agent performs actions in a logically incorrect sequence, where later actions depend on earlier actions that were not performed, or the sequence violates the workflow needed to interact with the environment.",
                        "example": "Task in a web automation environment: 'On the site, search for 'wireless mouse', filter price <= $25, open the first result, add to cart.' The agent clicks 'Add to cart' before applying the price filter and before opening the product page, triggering an error or adding an unintended item. The action order is logically incorrect for the UI flow.",
                    },
                    "UnsafeOrForbiddenAction": {
                        "definition": "The agent attempts an action that is disallowed by safety policies, permissions, or explicit environment rules (e.g., prohibited content or restricted operations). Trigger this error only when the environment rules or safety policy explicitly make the action non-executable or the action is rejected or blocked (e.g., denial message, permission error, policy refusal), not merely because the action is undesirable.",
                        "example": "Task in a web automation environment: 'On the site, search for 'wireless mouse', filter price <= $25, open the first result, add to cart.' The agent attempts to access a restricted admin page; the environment returns '403 Forbidden / Access Denied' or a policy refusal, and the action cannot proceed.",
                    },
                },
            },
            "obs": {
                "guidance": "OBS focuses on reading and interpreting environment or tool feedback. OBS is not about making a poor decision given correctly understood evidence (that is REASON), and not about tool-call formatting (that is ACT). If the agent reads a field incorrectly (e.g., mis-parses '17' as '7'), that is MisreadOutput. If it grounds to the wrong field, entity, or element (correctly reading it, but it is the wrong target), that is GroundingFail.",
                "errors": {
                    "MisreadOutput": {
                        "definition": "The agent misinterprets the environment response or tool output, such as misunderstanding a field, reading an error message as a successful result, or confusing similar entities. The agent's internal state updates are based on a wrong reading.",
                        "example": "Task: 'Run a tool that returns JSON with fields {status, total, items[]} and extract total and the top item name.' The tool returns {status:'ok', total:17, items:[{name:'X'}]}, but the agent reads total as 7 or interprets status as the count, producing the wrong extracted value.",
                    },
                    "IgnoreOutput": {
                        "definition": "The agent fails to use critical information present in the current output or visible prior output, including error diagnostics, key facts, direct answers, completion signals, or constraints. The information is available in the observation stream but is not incorporated into subsequent reasoning or planning.",
                        "example": "Task: 'Run a tool that returns JSON with fields {status, total, items[]} and extract total and the top item name.' The tool output includes total: 17 and items[0].name: 'X', but the agent ignores these fields and continues querying or claims the tool did not provide totals. Or a prior tool output already contains the needed answer, but the agent overlooks it and searches elsewhere.",
                    },
                    "TimingIssue": {
                        "definition": "The agent's observation is invalid because it acts before the environment is ready (page not loaded, async result not returned, tool still processing). The agent treats partial or empty output as final.",
                        "example": "Task: 'Run a tool that returns JSON with fields {status, total, items[]} and extract total and the top item name.' The environment initially returns {status:'pending'} and later would return the final JSON, but the agent treats the pending response as final, concluding no items found and stopping early.",
                    },
                    "GroundingFail": {
                        "definition": "The agent fails to correctly ground its intent to the right entity, element, or field in the environment or tool output. This includes selecting or extracting the wrong target due to ambiguous names, similar UI elements, wrong page section, wrong JSON field mapping, or incorrect entity alignment across sources. The needed information exists and is accessible, but the agent binds to the wrong referent, so subsequent reasoning is built on the wrong object or value.",
                        "example": "Task: 'On Rotten Tomatoes, report the Critic Score for the TV series The Office (US) and cite the page URL.' The agent searches The Office Rotten Tomatoes and opens the page for The Office (UK) (same title, different entity). It then correctly reads Critic Score: 98 percent from that page and reports it as the score for The Office (US). The extraction is correct for the opened page, but the entity alignment is wrong, so the answer is grounded to the wrong show.",
                    },
                },
            },
            "verify": {
                "guidance": "VERIFY focuses on progress validation and task-completion self-checks.",
                "errors": {
                    "NoVerification": {
                        "definition": "The agent fails to perform a progress or goal-completion check before proceeding or finalizing. It does not explicitly confirm that all required task conditions are satisfied. This is a missing progress validation rather than a general lack of citations.",
                        "example": "Task: 'Find the company's 2023 revenue in USD and provide the source URL.' The agent performs searches and extracts a number. It extracts '$5.2B' from a page but never checks whether it is for 2023 (it is actually 2022), never checks the unit or currency, and does not confirm it has a usable source URL, then returns the answer as complete.",
                    },
                    "WrongVerification": {
                        "definition": "The agent attempts a progress or goal-completion check, but the verification criterion, target, or progress assessment is wrong. It validates the wrong requirement (wrong year, metric, currency, or entity), uses an invalid success condition, is too optimistic about unresolved work, is too pessimistic about already completed work, or misses that a requirement has already been satisfied.",
                        "example": "Task: 'Find the company's 2023 revenue in USD and provide the source URL.' The agent performs searches and extracts a number. The agent verifies by checking a page that lists net income and concludes it confirmed revenue, or it verifies '2023' using a different company with the same name, then marks progress as successfully verified. Conversely, after obtaining a valid 2023 USD revenue and source URL, it incorrectly marks the task as incomplete and keeps searching.",
                    },
                    "PrematureTermination": {
                        "definition": "The agent terminates the task (returns a final answer or declares completion) while progress validation indicates unresolved requirements remain, or without performing the required progress validation steps. The key is that the agent stops even though the task is not yet completed according to the stated constraints.",
                        "example": "Task: 'Find the company's 2023 revenue in USD and provide the source URL.' The agent finds a revenue number and a source, but the number is in EUR and the task requires USD. Instead of converting or finding a USD source, it ends and claims completion.",
                    },
                    "InfiniteRetry": {
                        "definition": "The agent repeatedly retries the same failing approach or loops through the same actions without incorporating new information or changing strategy. The retries are ineffective and do not increase success probability.",
                        "example": "Task: 'Find the company's 2023 revenue in USD and provide the source URL.' The agent repeatedly re-runs essentially the same search or extraction steps after failing to meet constraints (e.g., keeps fetching the same page that lacks 2023 revenue), without changing queries, sources, or strategy, and without improving satisfaction of requirements.",
                    },
                },
            },
            # v5 spec §4.1.5.4: environment-side anomaly axis. attribution = "env"
            # for any of these triggers; they run alongside cat-1/2/3 but never
            # count toward the agent failure chain.
            "env": {
                "guidance": "ENV triggers fire when the agent's tool call at this step is itself correct (right tool, right params, consistent with the established sub-goal), but the immediately following environment response is anomalous. Set attribution=\"env\" for these — they do NOT count toward the agent failure chain.",
                "errors": {
                    "AdOverlayHijack": {
                        "definition": "The search / page response is hijacked by an ad vignette / overlay, and the agent has no admissible action-space alternative.",
                        "example": "Browser tool returns a full-page ad overlay covering the requested page; the available actions list contains only 'close ad / scroll' but every selector that would dismiss the overlay is gone or no-op.",
                    },
                    "ContentFilterBlock": {
                        "definition": "The model API or platform content filter rejects a syntactically well-formed query that does not actually violate any visible policy.",
                        "example": "An LLM-based search tool returns 'I cannot help with that' on a benign factual lookup whose phrasing trips an over-broad safety filter.",
                    },
                    "ToolExtractorDegenerate": {
                        "definition": "The tool is structurally degenerate on this target domain: regardless of input, it returns a fixed literal / empty / repeated payload (e.g. SPA scrape returns only the page title).",
                        "example": "A web-scraping tool always returns just the <title> tag for any single-page-app URL because it doesn't render JavaScript, so the agent never gets the actual page content.",
                    },
                    "RateLimitOrTransient": {
                        "definition": "The observation contains verbatim evidence of HTTP 429 / 5xx / timeout / connection reset / service unavailable / rate limited.",
                        "example": "The tool response body literally contains '429 Too Many Requests' or 'Service Unavailable' / 'timed out'.",
                    },
                    "EmptyOrRepeatedPayload": {
                        "definition": "The tool returns structurally empty / constant / payload identical to a prior call, with no clear structural reason (cache, jitter, pagination, etc.).",
                        "example": "Every call to the same search tool with different queries returns the exact same one-item list with the same content.",
                    },
                },
            },
        }

    def get_module_definitions(self, module_name: str) -> Dict[str, Dict[str, str]]:
        """Get all error definitions for a specific module."""
        module = self.definitions.get(module_name.lower())
        if not module:
            return {}
        return module.get("errors", {})

    def get_module_guidance(self, module_name: str) -> str:
        """Get guidance text for a specific module."""
        module = self.definitions.get(module_name.lower())
        if not module:
            return ""
        return module.get("guidance", "")

    def format_for_phase1_prompt(self, module_name: str) -> str:
        """Format error definitions for Phase 1 prompt with examples."""
        module_key = module_name.lower()
        module = self.definitions.get(module_key)

        if not module:
            return f"No error definitions found for module: {module_name}"

        guidance = module.get("guidance", "")
        formatted = f"MODULE GUIDANCE ({module_key.upper()}):\n{guidance}\n\n"
        formatted += f"DETAILED ERROR TYPE DEFINITIONS FOR {module_key.upper()} MODULE:\n\n"

        for error_type, details in module.get("errors", {}).items():
            formatted += f"- {error_type}:\n"
            formatted += f"  Definition: {details['definition']}\n"
            if details.get("example"):
                formatted += f"  Example: {details['example']}\n"
            formatted += "\n"

        formatted += "- no_error: No error detected in this module\n"

        return formatted

    def format_for_phase2_prompt(self) -> str:
        """Format all error definitions for Phase 2 critical error identification (no examples)."""
        reference = "COMPLETE ERROR TYPE REFERENCE WITH DEFINITIONS:\n\n"

        for module_key in self.get_all_modules():
            module = self.definitions[module_key]
            reference += f"=== {module_key.upper()} MODULE ===\n"
            guidance = module.get("guidance")
            if guidance:
                reference += f"Guidance: {guidance}\n"
            for error_type, details in module.get("errors", {}).items():
                reference += f"- {error_type}: {details['definition']}\n"
            reference += "\n"

        return reference

    def get_valid_error_types(self, module_name: str) -> List[str]:
        """Get list of valid error type names for a module."""
        module = self.definitions.get(module_name.lower())
        if not module:
            return ["no_error"]
        return list(module.get("errors", {}).keys()) + ["no_error"]

    def get_all_modules(self) -> List[str]:
        """Get list of all modules in taxonomy order."""
        return list(self.definitions.keys())


_DEFAULT_LOADER: "ErrorDefinitionsLoader" = ErrorDefinitionsLoader()


def is_valid_full_error_type(full_type: str) -> bool:
    """Return ``True`` iff ``full_type`` is ``"<module>.<subtype>"`` from
    the canonical detector taxonomy.

    Used by ``data_processing/`` converters and by ``score_steps`` to
    validate ``critical_error_type`` strings.
    """
    if not isinstance(full_type, str) or "." not in full_type:
        return False
    module, _, subtype = full_type.partition(".")
    module = module.strip().lower()
    subtype = subtype.strip()
    if not module or not subtype:
        return False
    valid_subtypes = _DEFAULT_LOADER.get_valid_error_types(module)
    return subtype in valid_subtypes and subtype != "no_error"


def split_full_error_type(full_type: str) -> tuple:
    """Split ``"module.subtype"`` into ``(module, subtype)``. Returns
    ``(None, None)`` if the input is malformed."""
    if not isinstance(full_type, str) or "." not in full_type:
        return (None, None)
    module, _, subtype = full_type.partition(".")
    module = module.strip().lower() or None
    subtype = subtype.strip() or None
    return (module, subtype)
