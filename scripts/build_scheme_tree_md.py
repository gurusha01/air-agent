"""Generate a per-step strategy tree markdown for any reward scheme.

Uses the same clean windowing/pairing logic as analyze_all_schemes.py.
Usage: build_scheme_tree_md.py <scheme> where scheme in {v6,v7,v8,v9}.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze_all_schemes import SCHEMES, load_window, match_pairs, categorize, BASELINE, BATCH_SIZE, OUT_DIR


REWARD_DESCR = {
    "v6": "+1 s>b / 0 s<=b / -0.5 fault",
    "v7": "+1 s>0.88 / +0.2 b<=s<=0.88 / 0 s<b / -0.5 fault",
    "v8": "+1 s>=best_ever / +0.2 b<s<best_ever / 0 s<=b / -0.5 fault (end-of-step snapshot)",
    "v9": "+1 s>p70 / +0.2 b<s<=p70 / 0 s<=b / -0.5 fault (N=64 rolling window)",
}


def short(direction: str) -> str:
    d = (direction or "").strip().replace("\n", " ")
    d = re.sub(r"^(write|modify|update|create|implement)[^.]*?(that|to)[:]?\s*", "", d, flags=re.I)
    d = re.sub(r"\s+", " ", d)
    return d[:90]


def fmt_score(s):
    return "FAIL" if s is None else f"{s:.4f}"


def main(scheme):
    entry = next((s for s in SCHEMES if s[0] == scheme), None)
    if entry is None:
        raise SystemExit(f"Unknown scheme {scheme!r}")
    scheme_id, pretty, r_path, rw_path, start, end, scheme_tag = entry
    rollouts = load_window(r_path, start, end, scheme_tag="any")
    rewards = load_window(rw_path, start, end, scheme_tag=scheme_tag)
    if not rewards:
        print(f"No rewards found for {scheme}")
        return
    pairs, n_steps = match_pairs(rollouts, rewards)

    out = [f"# FashionMNIST {pretty} — Per-Step Strategy Tree\n"]
    out.append(f"- Baseline: **{BASELINE}**")
    out.append(f"- Reward: {REWARD_DESCR.get(scheme_id, '?')}")
    out.append(f"- Structure: each step runs 8 parallel rollouts; each rollout = root + 1 child.")
    out.append(f"- Steps logged: **{n_steps}** ({n_steps * BATCH_SIZE} episodes)\n---\n")

    for s in range(n_steps):
        step_pairs = pairs[s * BATCH_SIZE:(s + 1) * BATCH_SIZE]
        bests = []
        for rw, _ in step_pairs:
            best = BASELINE
            for _, sc in rw.get("tree", []):
                if sc is not None and sc > best:
                    best = sc
            bests.append(best)
        mean_r = sum(rw["reward"] for rw, _ in step_pairs) / BATCH_SIZE
        succ = sum(1 for b in bests if b > BASELINE)
        out.append(f"## Step {s}  ·  max={max(bests):.4f}  ·  mean_reward={mean_r:+.3f}  ·  success={succ}/{BATCH_SIZE}")
        out.append("```")
        out.append(f"root (baseline={BASELINE})")
        for i, (rw, rl) in enumerate(step_pairs):
            branch = "└──" if i == len(step_pairs) - 1 else "├──"
            cat = categorize((rl or {}).get("direction", "")) if rl else "?"
            label = short((rl or {}).get("direction", "(no match)"))
            tree = rw.get("tree", [])
            node_score = tree[1][1] if len(tree) > 1 else None
            fault = " [FAULT]" if (rl and rl.get("executor_fault")) else ""
            out.append(f" {branch} [{cat:8s}] acc={fmt_score(node_score)}  r={rw['reward']:+.2f}{fault}  │ {label}")
        out.append("```\n")

    out_path = OUT_DIR / f"fmnist_{scheme}_strategy_tree.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out))
    print(f"Wrote {out_path} ({n_steps} steps)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "v7")
