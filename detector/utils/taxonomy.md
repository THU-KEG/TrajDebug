### agent error taxonomy：

1. PLAN

  > ***Module guidance**：*
  >
  > **PLAN focuses on the agent’s multi-step strategy and task decomposition.** Do **not** judge low-level tool syntax here; if the plan is sound but execution fails due to a malformed tool call, that belongs to ACT.

  1.1 BadDecomposition: 分解不当

  > **Definition :**
  > The agent decomposes the task into sub-tasks incorrectly, such that key sub-tasks are missing, redundant, or wrongly specified. This includes omitting a required step (e.g., not checking a necessary constraint) or defining a subgoal that does not contribute to the final objective.
  >
  > **Example :**
  > Task: “Book a 30-minute meeting with Alex next week; avoid Tue; use the calendar tool; propose 3 options; include a Zoom link.”
  > The agent’s plan is: “(1) Ask Alex for availability, (2) schedule meeting.” It omits key sub-tasks: checking the user’s calendar conflicts, enforcing “avoid Tue,” generating 3 candidate slots, and adding a Zoom link.

  1.2 WrongOrder: 步骤顺序错误

  > **Definition :**
  > The agent chooses an incorrect order of steps such that later actions depend on earlier steps that were not completed, or conclusions are drawn before collecting necessary evidence. The workflow ordering prevents successful execution or produces invalid results.
  >
  > **Example :**
  >
  > Task: “Book a 30-minute meeting with Alex next week; avoid Tue; use the calendar tool; propose 3 options; include a Zoom link.”
  >
  > The agent immediately creates the calendar event before checking conflicts and before generating 3 options. After creation fails due to conflict/Tue restriction, it backtracks repeatedly. The ordering prevents smooth execution.

  1.3 UnrealisticPlan: 不现实计划

  > **Definition :**
  > The agent forms a plan that relies on non-existent tools, unavailable APIs, inaccessible websites, missing credentials/permissions, or information sources that the environment cannot provide. The plan is infeasible given the tools and constraints.
  >
  > **Example :**
  > Task: “Book a 30-minute meeting with Alex next week; avoid Tue; use the calendar tool; propose 3 options; include a Zoom link.”
  >
  > The agent plans to “query Alex’s private calendar directly via API” even though the environment only provides the user’s calendar tool and has no access to Alex’s calendar. The plan depends on an unavailable capability.

  1.4 GoalDrift: 跑题

  >  **Definition :**
  > The agent gradually shifts away from the original task objective or constraints, pursuing a different question, relaxing requirements (e.g., ignoring region, season count), or focusing on irrelevant details. The plan no longer targets the correct deliverable.
  >
  > **Example :**
  >
  > Task: “Book a 30-minute meeting with Alex next week; avoid Tue; use the calendar tool; propose 3 options; include a Zoom link.”
  >
  > Instead of proposing 3 time options next week (excluding Tue) and including a Zoom link, the agent starts negotiating meeting agenda details and sends a long email draft without producing time options or a scheduling action.

  1.5 OverExplore: 无必要的大量探索

  > **Definition :**
  >  The agent performs unnecessary exploration, excessive searching, or redundant branching that does not materially increase the probability of success, causing wasted steps, cost, or context. The exploration is disproportionate to the task needs.
  >
  > **Example :**
  > Task: “Book a 30-minute meeting with Alex next week; avoid Tue; use the calendar tool; propose 3 options; include a Zoom link.”
  >
  > After already finding 5 valid free slots and preparing 3 options, the agent continues scanning multiple weeks and exploring “best meeting practices,” consuming steps without improving the scheduling outcome.

2. REASON

  > ***Module guidance**：*
  >
  > **REASON focuses on single-step reasoning and decision making.** REASON errors are about *how the agent converts evidence into a decision* in this step, not about whether the agent used the right tool (ACT) or misread the output (OBS). If the evidence is read correctly but the conclusion is still wrong, that is REASON.

  2.1 WrongChoice: 有证据但被忽略或被复述错

  > **Definition :**
  >  The agent's conclusion is driven by evidence that IS visible in the step (task statement, policy, prior tool output, or prior step still shown in the history) but the agent IGNORES that evidence or QUOTES IT INCORRECTLY. Typical forms: violating a task/policy constraint that is stated verbatim; restating a prior numeric or categorical field with the wrong value; summarizing earlier steps in a way that drops or distorts a still-visible key detail; attributing a failure to an unrelated cause while an explicit error-message cue is right there in the output. The defining feature is that the supporting evidence is present and legible, yet the decision is built on a distorted or discarded version of it.
  >
  > **Example :**
  >
  > *Example 1.* Task: “Choose the cheapest shipping option that arrives by Friday. A: $12 Thu; B: $8 Sat; C: $10 Fri.” The agent picks option B because it is cheapest, ignoring the Friday constraint that is stated verbatim in the task.
  >
  > *Example 2.* The previous tool output shows `total=17`, but in the current step the agent writes “previous total was 7” and plans around 7; the original value is still visible in the prior step, yet the agent mis-quotes it.

  2.2 InvalidInference: 证据被正确引用但推论过度引申

  > **Definition :**
  >  The evidence is visible AND the agent cites it accurately in this step, yet the conclusion over-reaches what that evidence can support. Typical forms: inferring that an entire tool, site, or strategy is broken from a single failed call; concluding a general rule from one data point; jumping from “A was observed” to “therefore B must hold” without a supporting link. The agent reads the evidence correctly; the failure is in the logical leap from evidence to conclusion, not in what the evidence says.
  >
  > **Example :**
  >
  > *Example 1.* Task: “Choose the cheapest shipping option that arrives by Friday. A: $12 Thu; B: $8 Sat; C: $10 Fri.” The agent correctly lists all three prices and deadlines, then concludes “A is the cheapest Friday-arriving option”. The evidence is read correctly but the inference over-reaches, since C also satisfies Friday and is cheaper than A.
  >
  > *Example 2.* After a single query returns no results, the agent concludes the entire search service is down, even though one empty result does not support that.

  2.3 MissingAssumptionCheck: 无证据却下结论

  > **Definition :**
  >  The agent relies on a claimed fact that has NO supporting evidence in the current step or any visible prior step. Typical forms: stating that a prior step observed X when no such prior step exists in the available context; recalling a tool call or tool result that was never produced; reflecting on a failure that never happened; proceeding from memory or guess for a value the agent would have needed to re-retrieve but didn't. The defining feature is that the supporting evidence for the claim does not exist in the available context — the agent has not checked, retrieved, or verified it.
  >
  > **Example :**
  >
  > *Example 1.* Task: “Choose the cheapest shipping option that arrives by Friday.” The agent writes “Saturday delivery counts as arriving by Friday” and proceeds; it never checks the wording of the deadline — an unchecked assumption.
  >
  > *Example 2.* The agent's thought includes “Based on my earlier call to the search tool, no matching items were found, so I will switch to browsing”; but the trajectory shown so far contains no prior call to any search tool, so the recalled event does not exist in the available context, yet the agent plans as if it did.

3. ACT

  > ***Module guidance**：*
  >
  > **ACT focuses on actions and tool usage.** ACT errors are about *doing the right kind of operation in the right form*. Do not penalize an action as ACT if it is blocked only due to external service failures unrelated to the agent's behavior. If the action is disallowed and explicitly rejected/blocked by environment rules or safety policy, that is UnsafeOrForbiddenAction.

  3.1 WrongTool: 选错工具

  > **Definition :**
  >  The agent uses an inappropriate tool or action type for the task context. The chosen tool cannot produce the needed information or is mismatched to the environment requirements.
  >
  > **Example :**
  >
  > Task in a web automation environment: “On the site, search for ‘wireless mouse’, filter price ≤ $25, open the first result, add to cart.”
  >
  > Instead of using the browser/search tool to interact with the site, the agent writes a natural-language guess of products and prices and outputs a final recommendation without performing any site actions.

  3.2 ToolSchemaMismatch: 参数不符合 Schema

  > **Definition :**
  >  The agent issues a tool call with invalid formatting or parameters that do not match the expected schema (missing required fields, wrong types, malformed JSON). The tool call fails or cannot be parsed.
  >
  > **Example :**
  >
  > Task in a web automation environment: “On the site, search for ‘wireless mouse’, filter price ≤ $25, open the first result, add to cart.”
  >
  > The environment expects a tool call like `{"action":"click","selector":"#add-to-cart"}` but the agent outputs malformed JSON or missing fields (e.g., `{"selector":"#add-to-cart"}`), causing the tool call to be rejected.

  3.3 WrongActionSequence: 动作序列逻辑错误

  > **Definition :**
  >  The agent performs actions in a logically incorrect sequence, where later actions depend on earlier actions that were not performed, or the sequence violates the workflow needed to interact with the environment.
  >
  > **Example :**
  >
  > Task in a web automation environment: “On the site, search for ‘wireless mouse’, filter price ≤ $25, open the first result, add to cart.”
  >
  > The agent clicks “Add to cart” before applying the price filter and before opening the product page, triggering an error or adding an unintended item. The action order is logically incorrect for the UI flow.

  3.4 UnsafeOrForbiddenAction: 触发安全、权限或规则限制

  > **Definition :**
  > The agent attempts an action that is disallowed by safety policies, permissions, or explicit environment rules (e.g., prohibited content or restricted operations). **Trigger this error only when the environment rules/safety policy explicitly make the action non-executable or the action is rejected/blocked** (e.g., denial message, permission error, policy refusal), not merely because the action is “undesirable.”
  >
  > **Example :**
  >
  > Task in a web automation environment: “On the site, search for ‘wireless mouse’, filter price ≤ $25, open the first result, add to cart.”
  >
  > The agent attempts to access a restricted admin page; the environment returns “403 Forbidden / Access Denied” or a policy refusal, and the action cannot proceed.

4. OBS

  > ***Module guidance**：*
  >
  > **OBS focuses on reading and interpreting environment/tool feedback.** OBS is not about making a poor decision given correctly understood evidence (that is REASON), and not about tool-call formatting (that is ACT). If the agent reads a field incorrectly (e.g., mis-parses “17” as “7”), that is MisreadOutput. If it grounds to the wrong field/entity/element (correctly reading it, but it’s the wrong target), that is GroundingFail.

  4.1 MisreadOutput: 读错返回内容

  > **Definition :**
  > The agent misinterprets the environment response or tool output, such as misunderstanding a field, reading an error message as a successful result, or confusing similar entities. The agent’s state updates are based on a wrong reading.
  >
  > **Example :**
  >
  > Task: “Run a tool that returns JSON with fields `{status, total, items[]}` and extract `total` and the top item name.”
  >
  > The tool returns `{"status":"ok","total":17,"items":[{"name":"X"}]}`, but the agent reads `total` as 7 or interprets `status` as the count, producing the wrong extracted value.

  4.2 IgnoreOutput: 忽略关键返回信息

  > **Definition :**
  > The agent fails to use critical information present in the current output or visible prior output, including error diagnostics, key facts, direct answers, completion signals, or constraints. The information is available in the observation stream but not incorporated into subsequent reasoning/planning.
  >
  > **Example :**
  >
  > Task: “Run a tool that returns JSON with fields `{status, total, items[]}` and extract `total` and the top item name.”
  >
  > The tool output includes `total: 17` and `items[0].name: "X"`, but the agent ignores these fields and continues querying or claims “the tool did not provide totals,” failing to use available information. Or a prior tool output already contains the needed answer, but the agent overlooks it and searches elsewhere.

  4.3 TimingIssue: 时序问题

  > **Definition :**
  >  The agent’s observation is invalid because it acts before the environment is ready (page not loaded, async result not returned, tool still processing). The agent treats partial/empty output as final.
  >
  > **Example :**
  >
  > Task: “Run a tool that returns JSON with fields `{status, total, items[]}` and extract `total` and the top item name.”
  >
  > The environment initially returns `{"status":"pending"}` and later would return the final JSON, but the agent treats the pending response as final, concluding “no items found” and stopping early.

  4.4 GroundingFail: 定位失败

  > **Definition :**
  > The agent fails to correctly ground its intent to the right entity/element/field in the environment or tool output. This includes selecting/extracting the wrong target due to ambiguous names, similar UI elements, wrong page section, wrong JSON field mapping, or incorrect entity alignment across sources. The key signature is: the needed information exists and is accessible, but the agent binds to the wrong referent, so subsequent reasoning is built on the wrong object/value.
  >
  > **Example :**
  >
  > Task: “On Rotten Tomatoes, report the *Critic Score* for the TV series ‘The Office (US)’ and cite the page URL.”
  >
  > The agent searches “The Office Rotten Tomatoes” and opens the Rotten Tomatoes page for ‘The Office (UK)’ (same title, different entity). It then correctly reads “Critic Score: 98%” from that page and reports it as the score for ‘The Office (US)’. The extraction is correct for the opened page, but the entity alignment is wrong, so the answer is grounded to the wrong show.

5. VERIFY

  > ***Module guidance**：*
  >
  > **VERIFY focuses on progress validation and task-completion self-checks.**

  5.1 NoVerification: 缺失验证

  > **Definition :**
  >
  > The agent fails to perform a progress/goal-completion check before proceeding or finalizing. It does not explicitly confirm that all required task conditions are satisfied . This is a *missing progress validation* rather than a general lack of citations.
  >
  > **Example :**
  >
  > Task: “Find the company’s 2023 revenue in USD and provide the source URL.” The agent performs searches and extracts a number.
  >
  > The agent extracts “$5.2B” from a page but never checks whether it is for 2023 (it is actually 2022), never checks the unit/currency, and does not confirm it has a usable source URL—then returns the answer as complete.

  5.2 WrongVerification: 验证方法错误

  > **Definition :**
  >
  > The agent attempts a progress/goal-completion check, but the verification criterion, target, or progress assessment is wrong. It validates the wrong requirement (wrong year/metric/currency/entity), uses an invalid success condition, is too optimistic about unresolved work, is too pessimistic about already completed work, or misses that a requirement has already been satisfied.
  >
  > **Example :**
  >
  > Task: “Find the company’s 2023 revenue in USD and provide the source URL.” The agent performs searches and extracts a number.
  >
  > The agent “verifies” by checking a page that lists *net income* and concludes it confirmed *revenue*, or it verifies “2023” using a different company with the same name, then marks progress as successfully verified. Conversely, after obtaining a valid 2023 USD revenue and source URL, it incorrectly marks the task as incomplete and keeps searching.

  5.3 PrematureTermination: 任务未完成即终止

  > **Definition :**
  >
  > The agent terminates the task (returns a final answer or declares completion) while progress validation indicates unresolved requirements remain, or without performing the required progress validation steps. The key is that the agent stops even though the task is not yet completed according to the stated constraints.
  >
  > **Example :**
  >
  > Task: “Find the company’s 2023 revenue in USD and provide the source URL.” The agent performs searches and extracts a number.
  >
  > The agent finds a revenue number and a source, but the number is in EUR and the task requires USD. Instead of converting or finding a USD source, it ends and claims completion.

  5.4 InfiniteRetry: 无效的重试或死循环反思

  > **Definition :**
  >  The agent repeatedly retries the same failing approach or loops through the same actions without incorporating new information or changing strategy. The retries are ineffective and do not increase success probability.
  >
  > **Example :**
  >
  > Task: “Find the company’s 2023 revenue in USD and provide the source URL.” The agent performs searches and extracts a number.
  >
  > The agent repeatedly re-runs essentially the same search/extraction steps after failing to meet constraints (e.g., keeps fetching the same page that lacks 2023 revenue), without changing queries, sources, or strategy, and without improving satisfaction of requirements.