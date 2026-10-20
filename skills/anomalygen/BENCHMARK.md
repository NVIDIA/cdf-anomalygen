# Skill Benchmark: anomalygen

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `anomalygen`
- Evaluation date: 2026-10-08
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 5 evaluation tasks (5 positive)
- Dataset digest: `sha256:17d86fc72cefe39a08619c7b1bec68c403d32debe3d0ad73cb4d729065ceab84` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 75.2% — baseline ran, but no comparable score was available; uplift unavailable | 70.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 80.0% → 100.0% (+20.0 points) |
| Correctness | 0.0% → 64.0% (+64.0 points) | 0.0% → 48.0% (+48.0 points) |
| Discoverability | 97.2% — baseline ran, but no comparable score was available; uplift unavailable | 92.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 6.5% → 29.8% (+23.3 points) | 9.8% → 23.8% (+14.0 points) |
| Efficiency | 84.8% — baseline ran, but no comparable score was available; uplift unavailable | 90.4% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 1,982,146 | 1,315,139 | +667,007 | +50.72% | skill 5/5; base 5/5 |
| claude-code | 1 | 408,153 | 248,241 | +159,912 | +64.42% | skill 1/1; base 1/1 |
| claude-code | 2 | 545,930 | 345,597 | +200,333 | +57.97% | skill 1/1; base 1/1 |
| claude-code | 3 | 265,046 | 154,648 | +110,398 | +71.39% | skill 1/1; base 1/1 |
| claude-code | 4 | 358,845 | 382,650 | -23,805 | -6.22% | skill 1/1; base 1/1 |
| claude-code | 5 | 404,172 | 184,003 | +220,169 | +119.66% | skill 1/1; base 1/1 |
| codex | All cases | 737,229 | 583,401 | +153,828 | +26.37% | skill 5/5; base 5/5 |
| codex | 1 | 99,804 | 132,893 | -33,089 | -24.90% | skill 1/1; base 1/1 |
| codex | 2 | 188,820 | 147,973 | +40,847 | +27.60% | skill 1/1; base 1/1 |
| codex | 3 | 87,627 | 85,543 | +2,084 | +2.44% | skill 1/1; base 1/1 |
| codex | 4 | 160,974 | 146,353 | +14,621 | +9.99% | skill 1/1; base 1/1 |
| codex | 5 | 200,004 | 70,639 | +129,365 | +183.14% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 2,719,375 | 1,898,540 | +820,835 | +43.24% | skill 10/10; base 10/10 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 38 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 5 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** QUALITY/quality_efficiency: Large skill (9849 tokens, recommended max <5000). Per agentskills.io, SKILL.md should be concise (~500 lines) — large skill bodies increase token cost after invocation; long or unfocused top-level descriptions can degrade agent routing accuracy (`skills/anomalygen/SKILL.md`)
- **MEDIUM** SCHEMA/line_count: SKILL.md has 640 lines (limit: 500) (`skills/anomalygen/SKILL.md`)
- **MEDIUM** SECURITY/Session Persistence (RA2): Rogue Agent: mkdir -p ~/.anomalygen && cp scripts/agentic_mask_generation_serving/endpoints.example.yaml ~/.anomalygen (`SKILL.md:99`)
- **MEDIUM** SECURITY/Whitespace Padding (P9): Prompt Injection: U+0020 x117 (`SKILL.md:582`)
- **MEDIUM** SECURITY/Whitespace Padding (P9): Prompt Injection: U+0020 x117 (`SKILL.md:639`)
- 33 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
