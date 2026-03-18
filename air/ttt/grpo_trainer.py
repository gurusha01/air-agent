"""GRPO trainer for the scientist model.

Step-level GRPO: at each search step, generate K scientist decisions,
execute in parallel, compute group advantages, update LoRA weights.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from air.ttt.rewards import compute_group_advantages, compute_reward
from air.ttt.scientist_env import ParallelScientistEnv, ScientistState


# ---------------------------------------------------------------------------
# Scientist model with LoRA
# ---------------------------------------------------------------------------

class ScientistModel:
    """Qwen + LoRA wrapper for generation and log-prob computation."""

    def __init__(
        self,
        model_name: str,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        device: str = "cuda",
        load_in_4bit: bool = False,
    ):
        self.device = device
        self.model_name = model_name

        print(f"Loading {model_name} for scientist LoRA training...")
        if load_in_4bit:
            print("  Using 4-bit quantization (QLoRA)")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                trust_remote_code=True,
                device_map={"": device},
            )
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            ).to(device)

        # Apply LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.base_model, lora_config)
        self.model.print_trainable_parameters()

        # Gradient checkpointing: trade compute for memory. Without this,
        # all intermediate activations are stored for backward — OOMs on
        # long prompts (tree view grows with each step).
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        print("  Gradient checkpointing enabled")

    def generate_k(
        self,
        prompt: str,
        K: int,
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
    ) -> list[str]:
        """Generate K completions for the same prompt."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        # Batch generate K completions
        input_ids = inputs["input_ids"].repeat(K, 1)
        attention_mask = inputs["attention_mask"].repeat(K, 1)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        completions = []
        for k in range(K):
            # Decode only the generated tokens (after the prompt)
            gen_ids = outputs[k][input_len:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            completions.append(text)

        return completions

    def generate_k_verbalized(
        self,
        prompt: str,
        K: int,
        temperature: float = 0.7,
        max_new_tokens: int = 3072,
    ) -> list[str]:
        """Generate K diverse strategies via verbalized sampling (single call).

        Asks the model to produce K different strategies in <response> tags,
        then wraps each into a scientist output format for downstream parsing.
        """
        import re as _re

        vs_prompt = prompt + (
            f"\n\nBefore committing to a single direction, brainstorm {K} "
            f"FUNDAMENTALLY DIFFERENT strategies (different algorithms, different "
            f"features, different approaches). Format each as:\n"
            f"<response><text>concrete strategy description</text>"
            f"<probability>0.XX</probability></response>\n\n"
            f"Then pick the best one for your DIRECTION below."
        )

        messages = [{"role": "user", "content": vs_prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        raw = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

        # Parse <response> blocks
        strategies = []
        for m in _re.finditer(r'<response>(.*?)</response>', raw, _re.DOTALL):
            block = m.group(1)
            tm = _re.search(r'<text>(.*?)</text>', block, _re.DOTALL)
            if tm:
                strategies.append(tm.group(1).strip())

        # Wrap each strategy into scientist output format
        completions = []
        for strat in strategies[:K]:
            formatted = (
                f"REASONING:\nVerbalized sampling strategy.\n\n"
                f"DIRECTION:\n{strat}\n\n"
                f"MODE: explore\n\n"
                f"HYPOTHESES_TESTED:\n- {strat[:100]}\n\n"
                f"PREDICTED_SCORE_RANGE: [0.0, 1.0]\n\n"
                f"MEMORY:\nNONE"
            )
            completions.append(formatted)

        # Pad if fewer than K parsed
        while len(completions) < K:
            completions.append(
                f"REASONING:\nFallback.\n\nDIRECTION:\n{raw[:300]}\n\n"
                f"MODE: explore\n\nMEMORY:\nNONE"
            )

        if strategies:
            print(f"    Parsed {len(strategies)} diverse strategies")
            for i, s in enumerate(strategies[:K]):
                print(f"      [{i}] {s[:80]}")

        return completions

    def compute_log_probs(
        self, prompt: str, completions: list[str],
    ) -> list[torch.Tensor]:
        """Compute per-token log-probs for each completion (with gradients).

        Memory-efficient: computes cross-entropy per-token directly instead of
        materializing the full (seq_len, vocab_size) log-softmax tensor.
        """
        messages = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer(prompt_text, return_tensors="pt")["input_ids"]
        prompt_len = prompt_ids.shape[1]

        log_probs_list = []
        for completion in completions:
            full_text = prompt_text + completion
            inputs = self.tokenizer(
                full_text, return_tensors="pt", truncation=True,
                max_length=prompt_len + 2048,
            ).to(self.device)

            outputs = self.model(**inputs)
            logits = outputs.logits  # (1, seq_len, vocab)

            # Shift: predict token t from logits at position t-1
            shift_logits = logits[:, :-1, :].squeeze(0)  # (seq_len-1, vocab)
            shift_labels = inputs["input_ids"][:, 1:].squeeze(0)  # (seq_len-1,)

            # Memory-efficient: compute log-prob of the actual token only
            # using cross_entropy (fused log_softmax + nll, never materializes
            # the full log_softmax tensor)
            per_token_loss = torch.nn.functional.cross_entropy(
                shift_logits, shift_labels, reduction="none",
            )
            # cross_entropy returns -log_prob, so negate
            token_log_probs = -per_token_loss

            # Only take log-probs for the completion tokens (after prompt)
            completion_log_probs = token_log_probs[prompt_len - 1:]
            log_probs_list.append(completion_log_probs)

            # Free memory between completions
            del outputs, logits, shift_logits, per_token_loss
            torch.cuda.empty_cache()

        return log_probs_list

    def compute_reference_log_probs(
        self, prompt: str, completions: list[str],
    ) -> list[torch.Tensor]:
        """Compute log-probs under the base (reference) model."""
        self.model.disable_adapter_layers()
        with torch.no_grad():
            ref_lps = self.compute_log_probs(prompt, completions)
        self.model.enable_adapter_layers()
        return [lp.detach() for lp in ref_lps]

    def save_lora(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"LoRA saved to {path}")

    def load_lora(self, path: str):
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(
            self.base_model, path,
        ).to(self.device)
        print(f"LoRA loaded from {path}")


# ---------------------------------------------------------------------------
# GRPO Trainer
# ---------------------------------------------------------------------------

class GRPOTrainer:
    """Step-level GRPO training for the scientist."""

    def __init__(
        self,
        scientist_model: ScientistModel,
        env: ParallelScientistEnv,
        K: int = 4,
        learning_rate: float = 1e-5,
        kl_coeff: float = 0.01,
        epsilon_greedy: float = 0.1,
        reward_weights: tuple[float, float, float] = (0.3, 0.5, 0.2),
        reward_mode: str = "granular",
        reward_epsilon: float = 0.0,
        use_verbalized_sampling: bool = False,
        no_train: bool = False,
        output_dir: str = "outputs/ttt_grpo",
        max_episodes: int = 100,
        steps_per_episode: int = 5,
        gradient_accumulation_steps: int = 4,
    ):
        self.model = scientist_model
        self.env = env
        self.K = K
        self.kl_coeff = kl_coeff
        self.epsilon_greedy = epsilon_greedy
        self.w_explore, self.w_exploit, self.w_memory = reward_weights
        self.reward_mode = reward_mode
        self.reward_epsilon = reward_epsilon
        self.use_verbalized_sampling = use_verbalized_sampling
        self.no_train = no_train
        self.output_dir = Path(output_dir)
        self.max_episodes = max_episodes
        self.steps_per_episode = steps_per_episode
        self.grad_accum = gradient_accumulation_steps

        self.optimizer = torch.optim.AdamW(
            scientist_model.model.parameters(), lr=learning_rate,
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)

        # Live CSV for easy monitoring: tail -f rewards.csv
        self.csv_path = self.output_dir / "rewards.csv"
        with open(self.csv_path, "w") as f:
            f.write("episode,step,best_score,improvement,r_explore,r_exploit,"
                    "r_memory,r_total,loss,chosen_k,elapsed_s\n")

    def train(self) -> dict:
        """Main training loop."""
        all_stats = []

        for episode in range(self.max_episodes):
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1}/{self.max_episodes}")
            print(f"{'='*60}")

            ep_start = time.time()
            state = self.env.reset()
            ep_rewards = []
            ep_steps = []

            self.optimizer.zero_grad()

            for step_t in range(self.steps_per_episode):
                budget_left = self.steps_per_episode - step_t
                state = self.env.get_state(budget_left)
                prompt = state.prompt_text

                print(f"\n  Step {step_t + 1}/{self.steps_per_episode} "
                      f"(budget_left={budget_left})")

                # 1. Generate K scientist outputs
                if self.use_verbalized_sampling:
                    completions = self.model.generate_k_verbalized(
                        prompt, self.K, temperature=0.7,
                    )
                else:
                    completions = self.model.generate_k(
                        prompt, self.K, temperature=0.7,
                    )
                    print(f"    Generated {self.K} completions")

                # 2. Execute all K in parallel
                results = self.env.step_parallel(completions)

                # 3. Compute rewards
                rewards = []
                components_list = []
                for k in range(self.K):
                    info = results[k]
                    child = self.env.primary.nodes.get(
                        info["child_id"],
                        self.env.workers[k].nodes.get(info["child_id"]),
                    )
                    parent = self.env.primary.nodes[info["parent_id"]]
                    decision = info["parsed_decision"]

                    r, comps = compute_reward(
                        child=child,
                        parent=parent,
                        all_nodes=self.env.primary.nodes,
                        decision=decision,
                        baseline_score=self.env.primary.container.baseline_score,
                        higher_is_better=self.env.primary.task.higher_is_better,
                        w_explore=self.w_explore,
                        w_exploit=self.w_exploit,
                        w_memory=self.w_memory,
                        reward_mode=self.reward_mode,
                        epsilon=self.reward_epsilon,
                    )
                    rewards.append(r)
                    components_list.append(comps)

                # 4. Compute group advantages
                advantages = compute_group_advantages(rewards)

                scores_str = ", ".join(
                    f"{r:.3f}" for r in rewards
                )
                print(f"    Rewards: [{scores_str}]")
                print(f"    Advantages: [{', '.join(f'{a:.2f}' for a in advantages)}]")

                # 5. Compute loss and update (skip if no_train)
                total_loss_val = 0.0
                if not self.no_train:
                    ref_log_probs = self.model.compute_reference_log_probs(
                        prompt, completions,
                    )

                    for k in range(self.K):
                        lp_list = self.model.compute_log_probs(
                            prompt, [completions[k]],
                        )
                        lp = lp_list[0]

                        lp_sum = lp.sum()
                        kl = (lp - ref_log_probs[k]).mean()

                        loss_k = (-advantages[k] * lp_sum + self.kl_coeff * kl)
                        loss_k = loss_k / (self.K * self.grad_accum)

                        loss_k.backward()
                        total_loss_val += loss_k.item()

                        del lp, lp_sum, kl, loss_k, lp_list
                        torch.cuda.empty_cache()

                    print(f"    Loss: {total_loss_val * self.grad_accum:.4f}")

                    # 6. Gradient step
                    if (step_t + 1) % self.grad_accum == 0 or step_t == self.steps_per_episode - 1:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.model.parameters(), max_norm=1.0,
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        print(f"    [Gradient step applied]")
                else:
                    print(f"    [No training — search only]")

                # 7. Choose which child to continue from
                if random.random() < self.epsilon_greedy:
                    chosen_k = random.randint(0, self.K - 1)
                    print(f"    Continuing from random child k={chosen_k}")
                else:
                    # Pick highest reward
                    chosen_k = max(range(self.K), key=lambda k: rewards[k])
                    print(f"    Continuing from best child k={chosen_k} "
                          f"(reward={rewards[chosen_k]:.3f})")

                self.env.commit_child(chosen_k, results)

                # Log
                step_loss_val = total_loss_val * self.grad_accum
                ep_rewards.append(rewards)
                step_info = {
                    "step": step_t,
                    "rewards": rewards,
                    "advantages": advantages,
                    "components": components_list,
                    "chosen_k": chosen_k,
                    "loss": step_loss_val,
                    "scores": [r["score"] for r in results],
                    "statuses": [r["execution_status"] for r in results],
                }
                ep_steps.append(step_info)

                # Write live CSV row (flushes immediately for tail -f)
                best_so_far = max(
                    (n.score for n in self.env.primary.nodes.values()
                     if n.score is not None), default=0.0,
                )
                bl = self.env.primary.container.baseline_score
                mean_comps = {
                    k: sum(c.get(k, 0) for c in components_list) / max(len(components_list), 1)
                    for k in ("r_explore", "r_exploit", "r_memory", "total")
                }
                with open(self.csv_path, "a") as f:
                    f.write(f"{episode},{step_t},{best_so_far:.4f},"
                            f"{best_so_far - bl:.4f},"
                            f"{mean_comps['r_explore']:.4f},"
                            f"{mean_comps['r_exploit']:.4f},"
                            f"{mean_comps['r_memory']:.4f},"
                            f"{mean_comps['total']:.4f},"
                            f"{step_loss_val:.4f},"
                            f"{chosen_k},"
                            f"{time.time() - ep_start:.0f}\n")
                    f.flush()

            # Episode done
            elapsed = time.time() - ep_start
            best_score = max(
                (n.score for n in self.env.primary.nodes.values() if n.score is not None),
                default=0.0,
            )
            baseline = self.env.primary.container.baseline_score
            improvement = best_score - baseline

            ep_stat = {
                "episode": episode,
                "best_score": best_score,
                "baseline": baseline,
                "improvement": improvement,
                "elapsed_s": round(elapsed, 1),
                "steps": ep_steps,
                "total_nodes": len(self.env.primary.nodes),
            }
            all_stats.append(ep_stat)

            print(f"\n  Episode {episode + 1} done in {elapsed:.0f}s")
            print(f"  Baseline: {baseline:.4f} | Best: {best_score:.4f} | "
                  f"Improvement: {improvement:+.4f}")
            print(f"  Total nodes explored: {len(self.env.primary.nodes)}")

            # Save logs
            with open(self.output_dir / "logs" / f"episode_{episode}.json", "w") as f:
                json.dump(ep_stat, f, indent=2, default=str)

            # Checkpoint every 5 episodes
            if (episode + 1) % 5 == 0:
                self.model.save_lora(
                    str(self.output_dir / "checkpoints" / f"ep{episode + 1}")
                )

        # Save final
        self.model.save_lora(str(self.output_dir / "checkpoints" / "final"))

        summary = {
            "total_episodes": self.max_episodes,
            "stats": all_stats,
        }
        with open(self.output_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return summary
