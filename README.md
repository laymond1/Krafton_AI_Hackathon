# krafton-prelim-ops

Practical operations repo for a 4-hour online AI R&D hackathon preliminary round.

This repo is built for a human-controlled dual-system workflow:

- 1 Human Controller
- 1 Claude-based multi-agent system
- 1 Codex-based multi-agent system

It is intentionally lean. The goal is fast setup, low coordination overhead, easy side-by-side comparison, and stable final packaging under time pressure.

## Non-goals

- large research infrastructure
- general-purpose agent framework
- production orchestration system

## Operating model

### Human Controller

The human remains the control tower and final judge.

Responsibilities:

- read the problem directly
- decide priorities
- delegate work to Claude and Codex
- compare outputs
- resolve conflicts
- choose final direction
- approve or prepare the submission

### Claude system

Use Claude for reasoning-heavy work:

- problem interpretation
- requirement and constraint extraction
- strategy generation
- critical review
- explanation quality

Recommended roles:

- Parser
- Strategist
- Critic
- Writer

### Codex system

Use Codex for execution-heavy work:

- proceduralizing the task
- structuring working steps
- verification and packaging checks
- execution-oriented drafting
- lightweight scripts or scaffolding when useful

Recommended roles:

- Parser
- Builder
- Verifier
- Writer

## Design principles

- Lean over elaborate
- Comparison speed matters
- Facts and assumptions must be separated explicitly
- Submit-ready beats theoretically elegant
- AI assists, human judges
- Reuse process across Day 1 and Day 2, not problem content

## Standard output structure

Use this structure for major agent outputs unless the controller explicitly overrides:

```md
# Objective
# Facts from Problem
# Assumptions
# Proposed Approach
# Risks
# Deliverables
# Immediate Next Actions
```

## Repository layout

```text
krafton-prelim-ops/
├─ README.md
├─ prompts/
│  ├─ shared/
│  │  ├─ output_format.md
│  │  ├─ decision_protocol.md
│  │  └─ scoring_rubric.md
│  ├─ claude/
│  │  ├─ parser.md
│  │  ├─ strategist.md
│  │  ├─ critic.md
│  │  └─ writer.md
│  └─ codex/
│     ├─ parser.md
│     ├─ builder.md
│     ├─ verifier.md
│     └─ writer.md
├─ templates/
│  ├─ problem_brief.md
│  ├─ solution_brief.md
│  ├─ risk_checklist.md
│  ├─ experiment_log.md
│  ├─ final_submission.md
│  └─ comparison_table.md
├─ runs/
│  ├─ day1/
│  │  ├─ raw_problem/
│  │  ├─ claude/
│  │  ├─ codex/
│  │  ├─ merged/
│  │  └─ final/
│  └─ day2/
│     ├─ raw_problem/
│     ├─ claude/
│     ├─ codex/
│     ├─ merged/
│     └─ final/
├─ notes/
│  ├─ operator_log_day1.md
│  └─ operator_log_day2.md
├─ scripts/
│  ├─ init_run.sh
│  ├─ with_wslim.sh
│  ├─ make_worktrees.sh
│  ├─ sync_outputs.sh
│  ├─ diff_summary.sh
│  └─ package_submission.sh
├─ scratch/
│  ├─ quick_tests/
│  └─ temp_outputs/
└─ submissions/
   ├─ day1/
   └─ day2/
```

## Minimal workflow

1. Initialize the round directory:

   ```bash
   ./scripts/init_run.sh day1
   ```

2. Save the raw problem statement in `runs/day1/raw_problem/problem.md`.

3. Run both parser prompts against the same raw problem and save outputs into:

- `runs/day1/claude/parser.md`
- `runs/day1/codex/parser.md`

4. Run strategy and build passes, then critical review and verification passes.

5. Use `templates/comparison_table.md` and `prompts/shared/scoring_rubric.md` to compare outputs quickly.

6. Merge the chosen direction into `runs/day1/final/final_submission.md`.

7. Package the final result:

   ```bash
   ./scripts/package_submission.sh day1
   ```

Repeat the same process for `day2`, but keep Day 1 and Day 2 content logically separate.

## Runtime Environment

Default runtime for this repo is the Conda environment `wslim`.

Activate it directly:

```bash
source /mnt/sdab1/userHome/wslim/miniconda/etc/profile.d/conda.sh
conda activate wslim
```

Or use the repo wrapper:

```bash
./scripts/with_wslim.sh python --version
```

For Python and PyTorch commands in this repo, prefer the wrapper so execution stays on one consistent environment:

```bash
./scripts/with_wslim.sh python runs/day1/codex/submission_draft.py --smoke-test
```

## Recommended Git and worktree setup

Recommended branches:

- `main`
- `ops/claude`
- `ops/codex`
- `ops/final`

Recommended worktrees:

- `wt-claude`
- `wt-codex`
- `wt-final`

Purpose:

- isolate Claude outputs
- isolate Codex outputs
- keep a clean final integration area

Do not create worktrees per subagent. Do not create separate repositories per agent.

## Prompt usage

- shared prompts define format, decision rules, and scoring
- Claude prompts are tuned for reasoning-heavy tasks
- Codex prompts are tuned for execution-heavy tasks
- templates keep outputs comparable and easy to merge

## Ground rules

- facts from the problem must stay separate from assumptions
- unsupported claims should be removed or labeled
- lower-risk submit-ready output is usually better than ambitious but fragile output
- the human controller approves every important merge and the final submission
