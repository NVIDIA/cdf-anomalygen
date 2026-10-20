# Skill Benchmark: cdf-anomalygen-release

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `cdf-anomalygen-release`
- Evaluation date: 2026-10-20
- Evaluator version: `1.5.6`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`), Codex (`openai/openai/gpt-5.5`)
- Tasks: 3 evaluation tasks (3 positive)
- Dataset digest: `sha256:76dfb65a830d8ab5ade84c6cd8418b6df6f703d0825eb03add58ca9a529925c0` (skill-evaluator-dataset-snapshot/1)
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
| Overall | 88.1% — baseline ran, but no comparable score was available; uplift unavailable | 81.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 66.7% → 100.0% (+33.3 points) |
| Correctness | 0.0% → 93.3% (+93.3 points) | 66.7% → 86.7% (+20.0 points) |
| Discoverability | 100.0% — baseline ran, but no comparable score was available; uplift unavailable | 95.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 7.5% → 60.0% (+52.5 points) | 38.3% → 45.8% (+7.5 points) |
| Efficiency | 86.9% — baseline ran, but no comparable score was available; uplift unavailable | 79.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 468,763 | 788,913 | -320,150 | -40.58% | skill 3/3; base 3/3 |
| claude-code | 1 | 168,402 | 182,604 | -14,202 | -7.78% | skill 1/1; base 1/1 |
| claude-code | 2 | 237,502 | 516,325 | -278,823 | -54.00% | skill 1/1; base 1/1 |
| claude-code | 3 | 62,859 | 89,984 | -27,125 | -30.14% | skill 1/1; base 1/1 |
| codex | All cases | 194,595 | 1,123,937 | -929,342 | -82.69% | skill 3/3; base 3/3 |
| codex | 1 | 99,882 | 923,148 | -823,266 | -89.18% | skill 1/1; base 1/1 |
| codex | 2 | 47,144 | 129,025 | -81,881 | -63.46% | skill 1/1; base 1/1 |
| codex | 3 | 47,569 | 71,764 | -24,195 | -33.71% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 663,358 | 1,912,850 | -1,249,492 | -65.32% | skill 6/6; base 6/6 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. N/A means the relevant trajectory counters were not available; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 5 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 3 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SECURITY/Unknown (SDI-1): The manifest states the skill is 'Not for training or synthetic defect-image generation (SDG)' and the Purpose section r (`SKILL.md:160`)
- **MEDIUM** SECURITY/Unknown (RP1): MCP Rug Pull: Docker image references without a specific tag (:latest is implicit) or digest (@sha256:...) can be silently replaced by (`references/running.md:12`)
- **LOW** QUALITY/quality_discoverability: Description very long (216 chars, recommend 50-150) (`skills/cdf-anomalygen-release/SKILL.md`)
- **LOW** QUALITY/quality_reliability: No limitations documented (`skills/cdf-anomalygen-release/SKILL.md`)
- **LOW** SECURITY/Unknown (SDI-2): The manifest description limits the skill to building the chosen Docker target from source and running it. Step 2 adds ` (`SKILL.md:126`)

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
