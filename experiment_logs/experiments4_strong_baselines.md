# Experiment 4: Strong Model Baselines

## Motivation

We have LLM-Guided v2 results with GPT-4o as scientist + Qwen 4B as executor, showing strong scaling on BOS and mountaincar. Now we compare different model combinations to understand:
- Does a stronger executor (Claude Opus 4.6) improve AIRA MCTS?
- Does a stronger scientist (Claude Opus 4.6) improve LLM-Guided?
- Can an LLM-Guided approach with a strong model for BOTH roles outperform everything?
- How does Qwen 4B as scientist compare?

## 5 Experiment Suites

| # | Suite Name | Method | Scientist | Executor | Thinking Budget | Run Command |
|---|------------|--------|-----------|----------|----------------|-------------|
| 1 | `strong-aira-claude` | AIRA MCTS | N/A | Claude Opus 4.6 | 1024 | `--suite strong-aira-claude` |
| 2 | (existing `feb22`) | AIRA MCTS | N/A | Qwen 4B | N/A | Already have results |
| 3 | `strong-llm-guided-claude-both` | LLM-Guided v2 | Claude Opus 4.6 | Claude Opus 4.6 | 1024 (both) | `--suite strong-llm-guided-claude-both` |
| 4 | `strong-llm-guided-claude-scientist` | LLM-Guided v2 | Claude Opus 4.6 | Qwen 4B | 1024 (scientist) | `--suite strong-llm-guided-claude-scientist` |
| 5 | `strong-llm-guided-qwen-both` | LLM-Guided v2 | Qwen 4B | Qwen 4B | N/A | `--suite strong-llm-guided-qwen-both` |

Each suite: 4 tasks x 2 budgets (n5, n15) x 5 runs = 40 experiments.
Suite 2 already exists -> 160 new experiments total.

Meta-suite `strong-baselines` includes suites 1, 3, 4, 5 (all 160 new experiments).

## Tasks

| Task | Config | GPU | Max Actions |
|------|--------|-----|-------------|
| Titanic | `tasks/titanic.yaml` | No | 15 |
| House Price | `tasks/regressionKaggleHousePrice.yaml` | No | 15 |
| Battle of Sexes | `tasks/battleOfSexes.yaml` | No | 15 |
| Mountain Car | `tasks/rlMountainCarContinuous.yaml` | Yes | 20 |

## Models

- **Claude Opus 4.6**: `claude-opus-4-20250514` via `https://api.anthropic.com/v1/`
  - Thinking budget: 1024 tokens (minimum for extended thinking)
  - Uses ANTHROPIC_API_KEY from .env
- **Qwen 4B**: `Qwen/Qwen3-4B-Instruct-2507` via local vLLM (`http://localhost:8000/v1`)

## Output Directories

```
outputs/Strong_Baselines/
  aira_claude_opus/          # Suite 1
    {task}/aira_mcts_n{5,15}_r{1-5}/result.json
  llm_guided_claude_both/    # Suite 3
    {task}/llm_guided_n{5,15}_r{1-5}/result.json
  llm_guided_claude_scientist/ # Suite 4
    {task}/llm_guided_n{5,15}_r{1-5}/result.json
  llm_guided_qwen_both/      # Suite 5
    {task}/llm_guided_n{5,15}_r{1-5}/result.json
```

Suite 2 results are at `outputs/Feb22_Baselines/{task}/aira_mcts_n{5,15}_r{1-5}/result.json`.

## Execution Plan

Run sequentially to manage GPU + API rate limits:
1. Suite 5 (Qwen both) — all local, fastest
2. Suite 4 (Claude scientist + Qwen executor) — moderate API use
3. Suite 1 (AIRA Claude executor) — API for each node
4. Suite 3 (Claude both) — most API heavy

```bash
cd /home/ubuntu/MLScientist/MLGym

# Suite 5: Qwen both
uv run --project /home/ubuntu/MLScientist/air-agent \
    python /home/ubuntu/MLScientist/air-agent/air/run_parallel.py \
    --suite strong-llm-guided-qwen-both --gpus 1,2,3,4,5,6,7

# Suite 4: Claude scientist + Qwen executor
uv run --project /home/ubuntu/MLScientist/air-agent \
    python /home/ubuntu/MLScientist/air-agent/air/run_parallel.py \
    --suite strong-llm-guided-claude-scientist --gpus 1,2,3,4,5,6,7

# Suite 1: AIRA with Claude executor
uv run --project /home/ubuntu/MLScientist/air-agent \
    python /home/ubuntu/MLScientist/air-agent/air/run_parallel.py \
    --suite strong-aira-claude --gpus 1,2,3,4,5,6,7

# Suite 3: Claude both
uv run --project /home/ubuntu/MLScientist/air-agent \
    python /home/ubuntu/MLScientist/air-agent/air/run_parallel.py \
    --suite strong-llm-guided-claude-both --gpus 1,2,3,4,5,6,7
```

## Plotting

```bash
uv run --project /home/ubuntu/MLScientist/air-agent \
    python /home/ubuntu/MLScientist/air-agent/air/plot_strong_baselines.py
```

Output: `outputs/Strong_Baselines/strong_model_baselines.png`

## Verification

1. Smoke test: `--filter "titanic" --filter "n5_r1"` from each suite
2. Check thinking budget is being passed (look for longer responses from Claude)
3. Monitor API costs via Anthropic dashboard
