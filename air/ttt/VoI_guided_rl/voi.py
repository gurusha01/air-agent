"""
Value of Information (VoI) computation.

Measures how much resolving a hypothesis would change the scientist's
experiment proposals. Uses centroid cosine distance in embedding space.

VoI(H) = 1 - cosine_similarity(
    mean_embed(proposals | H unresolved),
    mean_embed(proposals | H resolved)
)
"""

from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


def embed_texts(
    texts: list[str],
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    max_length: int = 512,
) -> np.ndarray:
    """Get mean-pooled embeddings from the model's hidden states."""
    if not texts:
        return np.zeros((0, 1))

    embeddings = []
    model.eval()
    device = next(model.parameters()).device

    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=max_length, padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Use last hidden state, mean pool over tokens
            hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
            mask = inputs["attention_mask"].unsqueeze(-1)  # (1, seq_len, 1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (1, hidden_dim)
            embeddings.append(pooled.cpu().float().numpy()[0])

    return np.stack(embeddings)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_voi(
    proposals_unresolved: list[str],
    proposals_H_true: list[str],
    proposals_H_false: list[str],
    p_true: float,
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
) -> float:
    """Compute Value of Information using centroid cosine distance.

    Args:
        proposals_unresolved: K experiment proposals with current thought.md
        proposals_H_true: K proposals with hypothesis confirmed
        proposals_H_false: K proposals with hypothesis rejected
        p_true: P(H=true) from logit estimation
        model: the scientist model (for embeddings)
        tokenizer: the tokenizer

    Returns:
        VoI score in [0, 2] (0 = no information, higher = more informative)
    """
    if not proposals_unresolved or not proposals_H_true or not proposals_H_false:
        return 0.0

    # Embed all proposals
    emb_u = embed_texts(proposals_unresolved, model, tokenizer)
    emb_t = embed_texts(proposals_H_true, model, tokenizer)
    emb_f = embed_texts(proposals_H_false, model, tokenizer)

    # Centroids
    centroid_u = emb_u.mean(axis=0)

    # Resolved centroid = weighted mix of H_true and H_false centroids
    centroid_t = emb_t.mean(axis=0)
    centroid_f = emb_f.mean(axis=0)
    centroid_r = p_true * centroid_t + (1 - p_true) * centroid_f

    # VoI = 1 - cosine_similarity (higher = more information gain)
    voi = 1.0 - cosine_similarity(centroid_u, centroid_r)

    return max(0.0, voi)


def estimate_p_true(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    thought_md: str,
    hypothesis: str,
    task_description: str,
) -> float:
    """Estimate P(H=true) from model logits.

    Prompts the model with thought.md + hypothesis and reads the logit
    probabilities for "true" vs "false" tokens. Non-gameable because
    the model doesn't control its own logits.
    """
    prompt = (
        f"Given the following analysis and problem description, "
        f"is this hypothesis true?\n\n"
        f"Analysis:\n{thought_md}\n\n"
        f"Problem: {task_description}\n\n"
        f"Hypothesis: {hypothesis}\n\n"
        f"Answer (true or false):"
    )

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # last token logits

    # Find token IDs for "true" and "false"
    # Try multiple variants
    true_tokens = tokenizer.encode(" true", add_special_tokens=False)
    false_tokens = tokenizer.encode(" false", add_special_tokens=False)

    if not true_tokens or not false_tokens:
        return 0.5  # fallback

    true_logit = logits[true_tokens[0]].item()
    false_logit = logits[false_tokens[0]].item()

    # Softmax over just these two
    max_logit = max(true_logit, false_logit)
    p_true = np.exp(true_logit - max_logit) / (
        np.exp(true_logit - max_logit) + np.exp(false_logit - max_logit)
    )

    return float(np.clip(p_true, 0.05, 0.95))


def sample_proposals(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    prompt: str,
    K: int = 32,
    max_new_tokens: int = 128,
    temperature: float = 0.9,
) -> list[str]:
    """Sample K experiment proposals from the scientist model."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)

    proposals = []
    for _ in range(K):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        # Decode only the generated part
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        if text:
            proposals.append(text)

    return proposals
