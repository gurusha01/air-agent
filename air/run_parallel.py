#!/usr/bin/env python3
"""Parallel experiment runner for adaptive tree search.

Runs multiple tree search experiments concurrently, managing a pool of GPUs.
All experiments share a single vLLM server (GPU 0). Each experiment gets a
Docker container assigned to a GPU from the pool.

GPU allocation:
  - GPU 0: vLLM inference server (always reserved)
  - GPUs 1-7: available for experiment containers
  - RL tasks (JAX training): need dedicated GPU per container
  - Tabular tasks (sklearn/xgboost): CPU-only, can share GPUs

Usage:
    # Run from MLGym directory:
    cd /home/ubuntu/MLScientist/MLGym

    # Run all RL experiments:
    uv run --project /home/ubuntu/MLScientist/air-agent \
        python /home/ubuntu/MLScientist/air-agent/air/run_parallel.py \
        --suite rl

    # Run titanic re-run with packages:
    uv run --project /home/ubuntu/MLScientist/air-agent \
        python /home/ubuntu/MLScientist/air-agent/air/run_parallel.py \
        --suite titanic-rerun

    # Run a custom set of experiments:
    uv run --project /home/ubuntu/MLScientist/air-agent \
        python /home/ubuntu/MLScientist/air-agent/air/run_parallel.py \
        --experiments experiments.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


SCRIPT = Path(__file__).parent / "adaptive_tree_search.py"
AIRA_SCRIPT = Path(__file__).parent / "aira_dojo" / "search.py"
LINEAR_SCRIPT = Path(__file__).parent / "linear_search.py"
LLM_GUIDED_SCRIPT = Path(__file__).parent / "llm_guided_tree_search.py"
BASE_OUT = Path("outputs/adaptive_search_v3")


# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    name: str                  # e.g. "ucb_c1.0.g"
    task_config: str           # e.g. "tasks/titanic.yaml"
    task_label: str            # e.g. "titanic" (for output dir)
    selection_strategy: str    # "ucb", "open-ended", "signals"
    context: str               # "parent" or "global"
    needs_gpu: bool = False    # True for RL tasks
    node_budget: int = 12
    initial_breadth: int = 3
    max_actions: int = 15
    extra_args: list[str] = field(default_factory=list)
    is_aira: bool = False      # True for aira-dojo experiments
    is_linear: bool = False    # True for linear search (no branching)
    model: str = ""            # "" = use script default (local vLLM)
    vllm_url: str | None = None  # None = don't pass; "" = OpenAI API
    temperature: float = 0.0  # 0 = don't pass (use script default)
    reflexion: bool = False    # --reflexion / --no-reflexion
    base_output: str = ""      # override BASE_OUT (empty = use default)
    is_llm_guided: bool = False   # True for LLM-guided tree search (Exp 4)
    scientist_model: str = ""     # e.g. "gpt-4o" (only for is_llm_guided=True)


# ---------------------------------------------------------------------------
# Predefined experiment suites
# ---------------------------------------------------------------------------

def _make_configs(task_config: str, task_label: str, needs_gpu: bool,
                  node_budget: int = 12, initial_breadth: int = 3,
                  max_actions: int = 15) -> list[ExperimentConfig]:
    """Generate the standard set of 15 experiment configs for a task."""
    configs = []

    # UCB with parent context
    for c in [1.0, 1.41, 2.0]:
        configs.append(ExperimentConfig(
            name=f"ucb_c{c:.2g}.p", task_config=task_config,
            task_label=task_label, selection_strategy="ucb",
            context="parent", needs_gpu=needs_gpu,
            node_budget=node_budget, initial_breadth=initial_breadth,
            max_actions=max_actions,
            extra_args=["--ucb-c", str(c)],
        ))

    # UCB with global context
    for c in [1.0, 1.41, 2.0]:
        configs.append(ExperimentConfig(
            name=f"ucb_c{c:.02g}.g", task_config=task_config,
            task_label=task_label, selection_strategy="ucb",
            context="global", needs_gpu=needs_gpu,
            node_budget=node_budget, initial_breadth=initial_breadth,
            max_actions=max_actions,
            extra_args=["--ucb-c", str(c)],
        ))

    # Open-ended with parent context
    for tw, k in [(0.3, 2), (0.5, 2), (0.5, 3), (1.0, 2)]:
        configs.append(ExperimentConfig(
            name=f"oe_t{tw}_k{k}.p", task_config=task_config,
            task_label=task_label, selection_strategy="open-ended",
            context="parent", needs_gpu=needs_gpu,
            node_budget=node_budget, initial_breadth=initial_breadth,
            max_actions=max_actions,
            extra_args=["--ucb-c", "1.41", "--trend-weight", str(tw),
                        "--commitment-threshold", str(k)],
        ))

    # Open-ended with global context
    for tw, k in [(0.5, 2), (1.0, 2)]:
        configs.append(ExperimentConfig(
            name=f"oe_t{tw}_k{k}.g", task_config=task_config,
            task_label=task_label, selection_strategy="open-ended",
            context="global", needs_gpu=needs_gpu,
            node_budget=node_budget, initial_breadth=initial_breadth,
            max_actions=max_actions,
            extra_args=["--ucb-c", "1.41", "--trend-weight", str(tw),
                        "--commitment-threshold", str(k)],
        ))

    # Signal-based baselines
    configs.append(ExperimentConfig(
        name="regret.p", task_config=task_config,
        task_label=task_label, selection_strategy="signals",
        context="parent", needs_gpu=needs_gpu,
        node_budget=node_budget, initial_breadth=initial_breadth,
        max_actions=max_actions,
        extra_args=["--use-regret"],
    ))
    configs.append(ExperimentConfig(
        name="regret.g", task_config=task_config,
        task_label=task_label, selection_strategy="signals",
        context="global", needs_gpu=needs_gpu,
        node_budget=node_budget, initial_breadth=initial_breadth,
        max_actions=max_actions,
        extra_args=["--use-regret"],
    ))
    configs.append(ExperimentConfig(
        name="regret_depth.g", task_config=task_config,
        task_label=task_label, selection_strategy="signals",
        context="global", needs_gpu=needs_gpu,
        node_budget=node_budget, initial_breadth=initial_breadth,
        max_actions=max_actions,
        extra_args=["--use-regret", "--use-depth"],
    ))

    return configs


def _make_aira_configs(task_config: str, task_label: str, needs_gpu: bool,
                       node_budget: int = 12, max_actions: int = 15,
                       ) -> list[ExperimentConfig]:
    """Generate AIRA-dojo experiment configs (greedy, mcts, evo) for a task."""
    configs = []

    # Greedy (paper defaults: num_drafts=5, debug_prob=1.0)
    configs.append(ExperimentConfig(
        name="aira_greedy.g", task_config=task_config,
        task_label=task_label, selection_strategy="greedy",
        context="global", needs_gpu=needs_gpu, is_aira=True,
        node_budget=node_budget, max_actions=max_actions,
        extra_args=["--num-drafts", "5", "--debug-prob", "1.0"],
    ))

    # MCTS (paper defaults: uct_c=0.25, num_children=5)
    configs.append(ExperimentConfig(
        name="aira_mcts.g", task_config=task_config,
        task_label=task_label, selection_strategy="mcts",
        context="global", needs_gpu=needs_gpu, is_aira=True,
        node_budget=node_budget, max_actions=max_actions,
        extra_args=["--uct-c", "0.25", "--num-children", "5"],
    ))

    # Evolutionary (paper defaults: crossover_prob=0.5, crossover_gen=2)
    configs.append(ExperimentConfig(
        name="aira_evo.g", task_config=task_config,
        task_label=task_label, selection_strategy="evolutionary",
        context="global", needs_gpu=needs_gpu, is_aira=True,
        node_budget=node_budget, max_actions=max_actions,
        extra_args=["--crossover-prob", "0.5", "--crossover-gen", "2",
                     "--indiv-per-gen", "5"],
    ))

    return configs


def _make_api_model_configs(
    model: str,
    prefix: str,
    temperature: float,
    task_config: str,
    task_label: str,
    needs_gpu: bool,
    node_budget: int = 12,
    max_actions: int = 15,
) -> list[ExperimentConfig]:
    """Generate 6 experiment configs for an API model on a single task.

    Methods: linear, UCB, open-ended, aira_greedy, aira_mcts, aira_evo.
    """
    common = dict(
        task_config=task_config,
        task_label=task_label,
        needs_gpu=needs_gpu,
        model=model,
        vllm_url="",           # empty string = OpenAI API
        temperature=temperature,
    )
    linear_budget = node_budget * max_actions

    configs = [
        # 1. Linear search (single trajectory)
        ExperimentConfig(
            name=f"{prefix}_linear",
            selection_strategy="linear",
            context="global",
            is_linear=True,
            node_budget=node_budget,
            max_actions=linear_budget,
            **common,
        ),
        # 2. UCB tree search (c=1.0, global context)
        ExperimentConfig(
            name=f"{prefix}_ucb.g",
            selection_strategy="ucb",
            context="global",
            node_budget=node_budget,
            initial_breadth=3,
            max_actions=max_actions,
            extra_args=["--ucb-c", "1.0"],
            **common,
        ),
        # 3. Open-ended tree search
        ExperimentConfig(
            name=f"{prefix}_oe.g",
            selection_strategy="open-ended",
            context="global",
            node_budget=node_budget,
            initial_breadth=3,
            max_actions=max_actions,
            extra_args=["--ucb-c", "1.41", "--trend-weight", "0.5",
                        "--commitment-threshold", "2"],
            **common,
        ),
        # 4. AIRA-dojo Greedy
        ExperimentConfig(
            name=f"{prefix}_aira_greedy.g",
            selection_strategy="greedy",
            context="global",
            is_aira=True,
            node_budget=node_budget,
            max_actions=max_actions,
            extra_args=["--num-drafts", "5", "--debug-prob", "1.0"],
            **common,
        ),
        # 5. AIRA-dojo MCTS
        ExperimentConfig(
            name=f"{prefix}_aira_mcts.g",
            selection_strategy="mcts",
            context="global",
            is_aira=True,
            node_budget=node_budget,
            max_actions=max_actions,
            extra_args=["--uct-c", "0.25", "--num-children", "5"],
            **common,
        ),
        # 6. AIRA-dojo Evolutionary
        ExperimentConfig(
            name=f"{prefix}_aira_evo.g",
            selection_strategy="evolutionary",
            context="global",
            is_aira=True,
            node_budget=node_budget,
            max_actions=max_actions,
            extra_args=["--crossover-prob", "0.5", "--crossover-gen", "2",
                        "--indiv-per-gen", "5"],
            **common,
        ),
    ]
    return configs


def _make_sweep_configs(
    model: str,
    model_label: str,
    vllm_url: str,
    temperature: float = 0.9,
) -> list[ExperimentConfig]:
    """Generate method x task x budget sweep configs for scaling experiments."""

    TASKS = [
        ("tasks/titanic.yaml",                   "titanic",     False, 15),
        ("tasks/regressionKaggleHousePrice.yaml", "houseprice",  False, 15),
        ("tasks/battleOfSexes.yaml",              "bos",         False, 15),
        ("tasks/rlMountainCarContinuous.yaml",    "mountaincar", True,  20),
    ]
    BUDGETS = [5, 10, 15, 30, 60]

    configs: list[ExperimentConfig] = []
    common_model = dict(model=model, vllm_url=vllm_url, temperature=temperature)

    for task_config, task_label, needs_gpu, max_actions in TASKS:
        for budget in BUDGETS:
            # (a) AIRA MCTS
            configs.append(ExperimentConfig(
                name=f"{model_label}_aira_mcts_n{budget}",
                task_config=task_config, task_label=task_label,
                selection_strategy="mcts", context="global",
                needs_gpu=needs_gpu, is_aira=True,
                node_budget=budget, max_actions=max_actions,
                reflexion=False,
                extra_args=["--uct-c", "0.25", "--num-children", "5"],
                **common_model,
            ))

            # (b) UCB + VS + global
            configs.append(ExperimentConfig(
                name=f"{model_label}_ucb_g_n{budget}",
                task_config=task_config, task_label=task_label,
                selection_strategy="ucb", context="global",
                needs_gpu=needs_gpu,
                node_budget=budget, initial_breadth=3, max_actions=max_actions,
                reflexion=False,
                extra_args=["--ucb-c", "1.41"],
                **common_model,
            ))

            # (c) OE + global
            configs.append(ExperimentConfig(
                name=f"{model_label}_oe_g_n{budget}",
                task_config=task_config, task_label=task_label,
                selection_strategy="open-ended", context="global",
                needs_gpu=needs_gpu,
                node_budget=budget, initial_breadth=3, max_actions=max_actions,
                reflexion=False,
                extra_args=["--trend-weight", "0.5", "--commitment-threshold", "2"],
                **common_model,
            ))

            # (d) OE + reflexion
            configs.append(ExperimentConfig(
                name=f"{model_label}_oe_refl_n{budget}",
                task_config=task_config, task_label=task_label,
                selection_strategy="open-ended", context="global",
                needs_gpu=needs_gpu,
                node_budget=budget, initial_breadth=3, max_actions=max_actions,
                reflexion=True,
                extra_args=["--trend-weight", "0.5", "--commitment-threshold", "2"],
                **common_model,
            ))

            # (e) OE + 5 initial drafts
            configs.append(ExperimentConfig(
                name=f"{model_label}_oe_5draft_n{budget}",
                task_config=task_config, task_label=task_label,
                selection_strategy="open-ended", context="global",
                needs_gpu=needs_gpu,
                node_budget=budget, initial_breadth=5, max_actions=max_actions,
                reflexion=False,
                extra_args=["--trend-weight", "0.5", "--commitment-threshold", "2"],
                **common_model,
            ))

    return configs


SUITES: dict[str, list[ExperimentConfig]] = {}


def _build_suites():
    """Build predefined suites lazily."""
    if SUITES:
        return

    # RL tasks v1 — old prompts (model tries to rewrite Python code, fails)
    SUITES["rl"] = (
        _make_configs("tasks/rlMountainCarContinuous.yaml", "mountaincar",
                      needs_gpu=True, node_budget=8, initial_breadth=2,
                      max_actions=10)
        + _make_configs("tasks/rlMetaMaze.yaml", "metamaze",
                        needs_gpu=True, node_budget=8, initial_breadth=2,
                        max_actions=10)
    )

    # RL tasks v2 — config-only prompts, more actions for inspect+write+train+validate
    SUITES["rl-v2"] = (
        _make_configs("tasks/rlMountainCarContinuous.yaml", "mountaincar_v2",
                      needs_gpu=True, node_budget=8, initial_breadth=2,
                      max_actions=15)
        + _make_configs("tasks/rlMetaMaze.yaml", "metamaze_v2",
                        needs_gpu=True, node_budget=8, initial_breadth=2,
                        max_actions=15)
    )

    # RL tasks v3 — unrestricted prompts + fixed TFP deps + mandatory code reading
    SUITES["rl-v3"] = (
        _make_configs("tasks/rlMountainCarContinuous.yaml", "mountaincar",
                      needs_gpu=True, node_budget=8, initial_breadth=2,
                      max_actions=20)
        + _make_configs("tasks/rlMetaMaze.yaml", "metamaze",
                        needs_gpu=True, node_budget=8, initial_breadth=2,
                        max_actions=20)
    )

    # Titanic re-run with package fix
    SUITES["titanic-rerun"] = _make_configs(
        "tasks/titanic.yaml", "titanic_v2", needs_gpu=False,
    )

    # All tasks
    SUITES["all"] = SUITES["rl-v3"] + SUITES["titanic-rerun"]

    # AIRA-dojo suites
    SUITES["aira-titanic"] = _make_aira_configs(
        "tasks/titanic.yaml", "titanic", needs_gpu=False,
        node_budget=12, max_actions=15,
    )
    SUITES["aira-houseprice"] = _make_aira_configs(
        "tasks/regressionKaggleHousePrice.yaml", "houseprice",
        needs_gpu=False, node_budget=12, max_actions=15,
    )
    SUITES["aira-bos"] = _make_aira_configs(
        "tasks/battleOfSexes.yaml", "battleofsexes",
        needs_gpu=False, node_budget=12, max_actions=15,
    )
    SUITES["aira-mountaincar"] = _make_aira_configs(
        "tasks/rlMountainCarContinuous.yaml", "mountaincar",
        needs_gpu=True, node_budget=8, max_actions=20,
    )
    SUITES["aira-rl"] = (
        SUITES["aira-mountaincar"]
        + _make_aira_configs(
            "tasks/rlMetaMaze.yaml", "metamaze",
            needs_gpu=True, node_budget=8, max_actions=20,
        )
    )
    SUITES["aira-all"] = (
        SUITES["aira-titanic"]
        + SUITES["aira-houseprice"]
        + SUITES["aira-bos"]
        + SUITES["aira-rl"]
    )

    # --- API model suites (GPT-4o, o3) ---
    _TASK_SPECS = [
        ("tasks/titanic.yaml", "titanic", False, 12, 15),
        ("tasks/regressionKaggleHousePrice.yaml", "houseprice", False, 12, 15),
        ("tasks/battleOfSexes.yaml", "battleofsexes", False, 12, 15),
        ("tasks/rlMountainCarContinuous.yaml", "mountaincar", True, 8, 20),
    ]
    _API_MODELS = [
        ("gpt-4o", "gpt4o", 0.7),
        ("o3", "o3", 0.0),   # temp=0 → omitted for reasoning models
    ]
    for model_name, prefix, temp in _API_MODELS:
        model_cfgs = []
        for tc, tl, gpu, nb, ma in _TASK_SPECS:
            task_cfgs = _make_api_model_configs(
                model=model_name, prefix=prefix, temperature=temp,
                task_config=tc, task_label=tl, needs_gpu=gpu,
                node_budget=nb, max_actions=ma,
            )
            SUITES[f"{prefix}-{tl}"] = task_cfgs
            model_cfgs.extend(task_cfgs)
        SUITES[f"{prefix}-all"] = model_cfgs

    SUITES["api-all"] = SUITES["gpt4o-all"] + SUITES["o3-all"]

    # --- Sweep suites (method × task × budget scaling experiments) ---
    SUITES["sweep-qwen3"] = _make_sweep_configs(
        model="Qwen/Qwen3-4B-Instruct-2507",
        model_label="q3",
        vllm_url="http://localhost:8000/v1",
    )
    SUITES["sweep-coder"] = _make_sweep_configs(
        model="Qwen/Qwen2.5-Coder-7B-Instruct",
        model_label="qc",
        vllm_url="http://localhost:8001/v1",
    )
    SUITES["sweep-all"] = SUITES["sweep-qwen3"] + SUITES["sweep-coder"]

    # --- Feb22 Baselines: softmax vs AIRA MCTS, 5 runs each ---
    FEB22_OUT = str(Path(__file__).parent.parent / "outputs" / "Feb22_Baselines")
    FEB22_TASKS = [
        ("tasks/titanic.yaml",                   "titanic",     False, 15),
        ("tasks/regressionKaggleHousePrice.yaml", "houseprice",  False, 15),
        ("tasks/battleOfSexes.yaml",              "bos",         False, 15),
        ("tasks/rlMountainCarContinuous.yaml",    "mountaincar", True,  20),
    ]
    FEB22_BUDGETS = [5, 15]
    FEB22_RUNS = 5
    feb22_cfgs: list[ExperimentConfig] = []
    common_feb22 = dict(
        model="Qwen/Qwen3-4B-Instruct-2507",
        vllm_url="http://localhost:8000/v1",
        temperature=0.9,
        base_output=FEB22_OUT,
    )
    for task_config, task_label, needs_gpu, max_actions in FEB22_TASKS:
        for budget in FEB22_BUDGETS:
            for run in range(1, FEB22_RUNS + 1):
                # (1) Softmax + sibling-aware + reflexion
                feb22_cfgs.append(ExperimentConfig(
                    name=f"softmax_n{budget}_r{run}",
                    task_config=task_config, task_label=task_label,
                    selection_strategy="softmax", context="global",
                    needs_gpu=needs_gpu,
                    node_budget=budget, initial_breadth=3, max_actions=max_actions,
                    reflexion=True,
                    extra_args=[
                        "--ucb-c", "1.41",
                        "--trend-weight", "0.5",
                        "--softmax-tau", "0.5",
                        "--visit-alpha", "0.5",
                    ],
                    **common_feb22,
                ))
                # (2) AIRA MCTS
                feb22_cfgs.append(ExperimentConfig(
                    name=f"aira_mcts_n{budget}_r{run}",
                    task_config=task_config, task_label=task_label,
                    selection_strategy="mcts", context="global",
                    needs_gpu=needs_gpu, is_aira=True,
                    node_budget=budget, max_actions=max_actions,
                    reflexion=False,
                    extra_args=["--uct-c", "0.25", "--num-children", "5"],
                    **common_feb22,
                ))
    SUITES["feb22"] = feb22_cfgs

    # --- Feb22 OE baseline (open-ended, no reflexion) ---
    feb22_oe_cfgs: list[ExperimentConfig] = []
    for task_config, task_label, needs_gpu, max_actions in FEB22_TASKS:
        for budget in FEB22_BUDGETS:
            for run in range(1, FEB22_RUNS + 1):
                feb22_oe_cfgs.append(ExperimentConfig(
                    name=f"oe_n{budget}_r{run}",
                    task_config=task_config, task_label=task_label,
                    selection_strategy="open-ended", context="global",
                    needs_gpu=needs_gpu,
                    node_budget=budget, initial_breadth=3, max_actions=max_actions,
                    reflexion=False,
                    extra_args=[
                        "--ucb-c", "1.41",
                        "--trend-weight", "0.5",
                        "--commitment-threshold", "2",
                    ],
                    **common_feb22,
                ))
    SUITES["feb22-oe"] = feb22_oe_cfgs

    # --- LLM-Guided Tree Search (Experiment 4) ---
    LLM_GUIDED_OUT = str(Path(__file__).parent.parent / "outputs" / "LLM_Guided_v2")
    LLM_GUIDED_TASKS = [
        ("tasks/titanic.yaml",                   "titanic",     False, 15),
        ("tasks/regressionKaggleHousePrice.yaml", "houseprice",  False, 15),
        ("tasks/battleOfSexes.yaml",              "bos",         False, 15),
        ("tasks/rlMountainCarContinuous.yaml",    "mountaincar", True,  20),
    ]
    LLM_GUIDED_BUDGETS = [5, 15]
    LLM_GUIDED_RUNS = 5
    llm_guided_cfgs: list[ExperimentConfig] = []
    common_llm_guided = dict(
        model="Qwen/Qwen3-4B-Instruct-2507",
        vllm_url="http://localhost:8000/v1",
        temperature=0.9,
        base_output=LLM_GUIDED_OUT,
    )
    for task_config, task_label, needs_gpu, max_actions in LLM_GUIDED_TASKS:
        for budget in LLM_GUIDED_BUDGETS:
            for run in range(1, LLM_GUIDED_RUNS + 1):
                llm_guided_cfgs.append(ExperimentConfig(
                    name=f"llm_guided_n{budget}_r{run}",
                    task_config=task_config, task_label=task_label,
                    selection_strategy="llm_guided", context="global",
                    needs_gpu=needs_gpu,
                    node_budget=budget, initial_breadth=3, max_actions=max_actions,
                    is_llm_guided=True,
                    scientist_model="gpt-4o",
                    **common_llm_guided,
                ))
    SUITES["llm-guided"] = llm_guided_cfgs


# ---------------------------------------------------------------------------
# GPU Pool
# ---------------------------------------------------------------------------

class GPUPool:
    """Thread-safe GPU allocation pool."""

    def __init__(self, gpu_ids: list[int]):
        self._lock = threading.Lock()
        self._available = list(gpu_ids)
        self._condition = threading.Condition(self._lock)

    def acquire(self, needs_dedicated: bool = True) -> int:
        """Get a GPU. Blocks if none available."""
        with self._condition:
            while not self._available:
                self._condition.wait()
            gpu = self._available.pop(0)
            return gpu

    def release(self, gpu_id: int):
        """Return a GPU to the pool."""
        with self._condition:
            self._available.append(gpu_id)
            self._condition.notify()

    @property
    def available_count(self) -> int:
        with self._lock:
            return len(self._available)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

_print_lock = threading.Lock()


def _log(msg: str):
    with _print_lock:
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)


def run_one_experiment(cfg: ExperimentConfig, gpu_pool: GPUPool,
                       dry_run: bool = False) -> dict:
    """Run a single experiment. Returns result dict."""
    base = Path(cfg.base_output) if cfg.base_output else BASE_OUT
    out_dir = base / cfg.task_label / cfg.name
    result_file = out_dir / "result.json"

    # Skip if already completed
    if result_file.exists():
        try:
            result = json.loads(result_file.read_text())
            _log(f"SKIP {cfg.task_label}/{cfg.name} (score={result.get('best_score', '?'):.4f})")
            return {"name": cfg.name, "task": cfg.task_label,
                    "status": "skipped", "score": result.get("best_score")}
        except Exception:
            pass

    # Acquire GPU
    gpu = gpu_pool.acquire(needs_dedicated=cfg.needs_gpu)
    _log(f"START {cfg.task_label}/{cfg.name} on GPU {gpu}")

    if cfg.is_llm_guided:
        cmd = [
            sys.executable, str(LLM_GUIDED_SCRIPT),
            "--task-config", cfg.task_config,
            "--output-dir", str(out_dir),
            "--node-budget", str(cfg.node_budget),
            "--initial-breadth", str(cfg.initial_breadth),
            "--max-actions", str(cfg.max_actions),
            "--env-gpu", str(gpu),
        ]
        if cfg.scientist_model:
            cmd.extend(["--scientist-model", cfg.scientist_model])
        cmd.extend(cfg.extra_args)
    elif cfg.is_linear:
        cmd = [
            sys.executable, str(LINEAR_SCRIPT),
            "--task-config", cfg.task_config,
            "--output-dir", str(out_dir),
            "--max-actions", str(cfg.max_actions),
            "--env-gpu", str(gpu),
        ] + cfg.extra_args
    elif cfg.is_aira:
        cmd = [
            sys.executable, str(AIRA_SCRIPT),
            "--task-config", cfg.task_config,
            "--output-dir", str(out_dir),
            "--node-budget", str(cfg.node_budget),
            "--max-actions", str(cfg.max_actions),
            "--search-policy", cfg.selection_strategy,
            "--env-gpu", str(gpu),
        ] + cfg.extra_args
    else:
        cmd = [
            sys.executable, str(SCRIPT),
            "--task-config", cfg.task_config,
            "--output-dir", str(out_dir),
            "--node-budget", str(cfg.node_budget),
            "--initial-breadth", str(cfg.initial_breadth),
            "--max-actions", str(cfg.max_actions),
            "--selection-strategy", cfg.selection_strategy,
            "--context", cfg.context,
            "--env-gpu", str(gpu),
        ] + cfg.extra_args

    # Inject API model parameters when set
    if cfg.model:
        cmd.extend(["--model", cfg.model])
    if cfg.vllm_url is not None:
        cmd.extend(["--vllm-url", cfg.vllm_url])
    if cfg.temperature > 0:
        cmd.extend(["--temperature", str(cfg.temperature)])

    # Reflexion flag (both adaptive_tree_search and aira support it)
    if not cfg.is_linear and not cfg.is_llm_guided:
        if cfg.reflexion:
            cmd.append("--reflexion")
        else:
            cmd.append("--no-reflexion")

    if dry_run:
        _log(f"  DRY RUN: {' '.join(cmd)}")
        gpu_pool.release(gpu)
        return {"name": cfg.name, "task": cfg.task_label,
                "status": "dry_run", "cmd": " ".join(cmd)}

    start_time = time.time()
    log_file = out_dir / "experiment.log"
    os.makedirs(out_dir, exist_ok=True)

    try:
        with open(log_file, "w") as lf:
            proc = subprocess.run(
                cmd, stdout=lf, stderr=subprocess.STDOUT,
                timeout=21600,  # 6 hour timeout
            )
        elapsed = time.time() - start_time

        # Read result
        if result_file.exists():
            result = json.loads(result_file.read_text())
            score = result.get("best_score", None)
            _log(f"DONE {cfg.task_label}/{cfg.name} "
                 f"score={score:.4f} ({elapsed:.0f}s) GPU {gpu}")
            return {"name": cfg.name, "task": cfg.task_label,
                    "status": "done", "score": score, "elapsed": elapsed}
        else:
            _log(f"FAIL {cfg.task_label}/{cfg.name} "
                 f"(no result.json, exit={proc.returncode}, {elapsed:.0f}s) GPU {gpu}")
            return {"name": cfg.name, "task": cfg.task_label,
                    "status": "failed", "returncode": proc.returncode,
                    "elapsed": elapsed}

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        _log(f"TIMEOUT {cfg.task_label}/{cfg.name} ({elapsed:.0f}s) GPU {gpu}")
        return {"name": cfg.name, "task": cfg.task_label,
                "status": "timeout", "elapsed": elapsed}
    except Exception as e:
        _log(f"ERROR {cfg.task_label}/{cfg.name}: {e}")
        return {"name": cfg.name, "task": cfg.task_label,
                "status": "error", "error": str(e)}
    finally:
        gpu_pool.release(gpu)


def main():
    parser = argparse.ArgumentParser(
        description="Run adaptive tree search experiments in parallel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--suite",
        help="Predefined experiment suite to run (use --list-suites to see all)",
    )
    parser.add_argument(
        "--gpus", default="1,2,3,4,5,6,7",
        help="Comma-separated GPU IDs for experiment containers (default: 1-7)",
    )
    parser.add_argument(
        "--max-parallel", type=int, default=0,
        help="Max parallel experiments (0 = match GPU count)",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running")
    parser.add_argument(
        "--filter", default="",
        help="Only run experiments whose name contains this substring",
    )
    parser.add_argument(
        "--task-filter", default="",
        help="Only run experiments for this task (e.g. 'mountaincar')",
    )
    parser.add_argument("--list-suites", action="store_true",
                        help="List all available suites and exit")
    args = parser.parse_args()

    _build_suites()

    if args.list_suites:
        for name, cfgs in sorted(SUITES.items()):
            print(f"  {name:25s} ({len(cfgs)} experiments)")
        return

    if args.suite:
        if args.suite not in SUITES:
            parser.error(f"Unknown suite '{args.suite}'. Use --list-suites to see options.")
        experiments = SUITES[args.suite]
    else:
        parser.error("--suite is required (use --list-suites to see options)")

    # Apply filters
    if args.filter:
        experiments = [e for e in experiments if args.filter in e.name]
    if args.task_filter:
        experiments = [e for e in experiments
                       if args.task_filter in e.task_label]

    if not experiments:
        print("No experiments to run after filtering.")
        return

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    max_parallel = args.max_parallel or len(gpu_ids)

    # For tabular tasks that don't need dedicated GPUs,
    # we can run more in parallel than we have GPUs
    gpu_needing = [e for e in experiments if e.needs_gpu]
    cpu_only = [e for e in experiments if not e.needs_gpu]

    if not gpu_needing:
        # All CPU tasks — duplicate GPU IDs to allow more parallelism
        # Each tabular container gets a GPU assigned but doesn't use it
        expanded_gpus = gpu_ids * ((max_parallel + len(gpu_ids) - 1) // len(gpu_ids))
        gpu_pool = GPUPool(expanded_gpus[:max_parallel])
    else:
        gpu_pool = GPUPool(gpu_ids)

    print(f"=" * 60)
    print(f"  Parallel Experiment Runner")
    print(f"  Suite: {args.suite}")
    print(f"  Experiments: {len(experiments)}")
    print(f"  GPUs: {gpu_ids}")
    print(f"  Max parallel: {max_parallel}")
    print(f"=" * 60)
    print()

    for i, exp in enumerate(experiments):
        print(f"  [{i+1:2d}] {exp.task_label}/{exp.name} "
              f"({exp.selection_strategy}, {exp.context}, "
              f"gpu={'yes' if exp.needs_gpu else 'no'})")
    print()

    results = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(run_one_experiment, exp, gpu_pool, args.dry_run): exp
            for exp in experiments
        }

        for future in as_completed(futures):
            exp = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                _log(f"EXCEPTION {exp.task_label}/{exp.name}: {e}")
                results.append({"name": exp.name, "task": exp.task_label,
                                "status": "exception", "error": str(e)})

    elapsed = time.time() - start

    # Print summary
    print()
    print(f"=" * 60)
    print(f"  RESULTS ({elapsed:.0f}s total)")
    print(f"=" * 60)

    # Group by task
    by_task: dict[str, list[dict]] = {}
    for r in results:
        by_task.setdefault(r["task"], []).append(r)

    for task, task_results in sorted(by_task.items()):
        print(f"\n  {task}:")
        for r in sorted(task_results, key=lambda x: -(x.get("score") or 0)):
            status = r["status"]
            score = r.get("score")
            elapsed_s = r.get("elapsed", 0)
            if score is not None:
                print(f"    {r['name']:25s} {score:8.4f}  ({status}, {elapsed_s:.0f}s)")
            else:
                print(f"    {r['name']:25s} {'N/A':>8s}  ({status})")

    # Save results
    summary_file = BASE_OUT / f"parallel_run_{args.suite}_{int(time.time())}.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(results, indent=2))
    print(f"\n  Results saved to {summary_file}")


if __name__ == "__main__":
    main()
