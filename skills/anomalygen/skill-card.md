## Description: <br>
Use when running the CDF AnomalyGen pipeline over a defect dataset — mask placement, fine-tuning, synthetic defect-image generation (SDG), evaluation, quality refinement, and pseudo-labeling — even when the user names only one stage. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers building industrial defect-detection datasets use this skill to fine-tune the frozen Cosmos base into an anomaly-generation model and to produce, score, refine, and pseudo-label synthetic defect images from clean images and defect masks. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Yes] <br>
**Credential Type(s):** [API key] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Agentic mask generation reference (Step 2, AMG)](references/agentic-mask-generation.md) <br>
- [Mask-placement reference (Step 2, AMP)](references/mask-placement.md) <br>
- [Fine-tuning reference (Step 3)](references/fine-tuning.md) <br>
- [Quality-refinement reference (Step 6)](references/quality-refinement.md) <br>
- [Inputs reference](references/inputs.md) <br>
- [Verification reference (per-step gates)](references/verification.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Files] <br>
**Output Format:** [Markdown with inline bash and YAML code blocks; pipeline artifacts on disk (testcase JSONL, checkpoints, generated images, KPI JSON, COCO annotations)] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Each step is complete only when its named verification gate passes on the on-disk artifact] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
5 evaluation tasks (5 positive), 1 attempt per task, each run in an isolated sandbox pod; evaluated with and without the skill for baseline comparison. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Is the answer correct? Final-answer correctness against the reference answer. <br>
- Discoverability: Was the right skill loaded when needed? <br>
- Effectiveness: Did the skill help complete the task? Equal-weight mean of goal completion and expected workflow adherence. <br>
- Efficiency: Did it avoid wasted tool calls and token usage? 50% tool-call productivity and 50% token efficiency. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `skill_execution`: Whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>
- `skill_efficiency`: Tool-call productivity. <br>
- `token_efficiency`: Actual uncached prompt plus completion usage. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 75.2% — baseline ran, but no comparable score was available; uplift unavailable | 70.8% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 100.0% (±0.0 points) | 80.0% → 100.0% (+20.0 points) |
| Correctness | 0.0% → 64.0% (+64.0 points) | 0.0% → 48.0% (+48.0 points) |
| Discoverability | 97.2% — baseline ran, but no comparable score was available; uplift unavailable | 92.0% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 6.5% → 29.8% (+23.3 points) | 9.8% → 23.8% (+14.0 points) |
| Efficiency | 84.8% — baseline ran, but no comparable score was available; uplift unavailable | 90.4% — baseline ran, but no comparable score was available; uplift unavailable |

## Skill Version(s): <br>
1.2.0 (source: frontmatter, pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
