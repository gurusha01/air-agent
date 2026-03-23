"""
thought.md — the scientist's living mental model.

Contains active hypotheses with confidence levels, structural claims,
experimental results, and open questions. Updated after every experiment.
Replaces the flat memory list from the original tree search.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum


class HypothesisStatus(str, Enum):
    PROPOSED = "proposed"       # Just introduced, not yet tested
    TESTING = "testing"         # Validate nodes being run
    VALIDATED = "validated"     # Enough evidence to consider true
    REJECTED = "rejected"       # Evidence says false
    ABANDONED = "abandoned"     # Ran out of validate budget without resolution


@dataclass
class Hypothesis:
    id: str                                    # e.g. "H1"
    claim: str                                 # e.g. "FamilySize is a useful latent variable"
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    confidence: float = 0.5                    # 0-1, updated externally from logits
    validate_budget: int = 3                   # max validate nodes for this hypothesis
    validate_used: int = 0
    evidence: list[str] = field(default_factory=list)  # results from validate/challenge nodes


@dataclass
class ThoughtDoc:
    """The scientist's structured mental model of the problem."""

    hypotheses: dict[str, Hypothesis] = field(default_factory=dict)
    structural_claims: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    _next_id: int = 1

    def add_hypothesis(self, claim: str, validate_budget: int = 3) -> str:
        """Add a new hypothesis. Returns the hypothesis ID."""
        hid = f"H{self._next_id}"
        self._next_id += 1
        self.hypotheses[hid] = Hypothesis(id=hid, claim=claim, validate_budget=validate_budget)
        return hid

    def get_unvalidated(self) -> list[Hypothesis]:
        """Get hypotheses that can still be validated."""
        return [
            h for h in self.hypotheses.values()
            if h.status in (HypothesisStatus.PROPOSED, HypothesisStatus.TESTING)
            and h.validate_used < h.validate_budget
        ]

    def get_validated(self) -> list[Hypothesis]:
        """Get hypotheses that have been validated (eligible for challenge)."""
        return [h for h in self.hypotheses.values() if h.status == HypothesisStatus.VALIDATED]

    def record_validate(self, hid: str, result: str, success: bool):
        """Record a validate experiment result."""
        h = self.hypotheses[hid]
        h.validate_used += 1
        h.evidence.append(result)
        h.status = HypothesisStatus.TESTING
        if success:
            h.status = HypothesisStatus.VALIDATED
        elif h.validate_used >= h.validate_budget:
            h.status = HypothesisStatus.ABANDONED
            h.confidence = max(0.1, h.confidence - 0.3)

    def record_challenge(self, hid: str, result: str, hypothesis_survives: bool):
        """Record a challenge experiment result."""
        h = self.hypotheses[hid]
        h.evidence.append(f"[challenge] {result}")
        if not hypothesis_survives:
            h.status = HypothesisStatus.REJECTED
            h.confidence = 0.1

    def add_structural_claim(self, claim: str):
        """Add a structural claim about the problem."""
        if claim not in self.structural_claims:
            self.structural_claims.append(claim)

    def add_open_question(self, question: str):
        if question not in self.open_questions:
            self.open_questions.append(question)

    def render(self) -> str:
        """Render thought.md as text for the scientist prompt."""
        lines = ["# thought.md\n"]

        # Active hypotheses
        lines.append("## Hypotheses")
        if not self.hypotheses:
            lines.append("(No hypotheses yet — the problem is unexplored.)\n")
        for h in self.hypotheses.values():
            status_icon = {
                HypothesisStatus.PROPOSED: "?",
                HypothesisStatus.TESTING: "~",
                HypothesisStatus.VALIDATED: "✓",
                HypothesisStatus.REJECTED: "✗",
                HypothesisStatus.ABANDONED: "—",
            }.get(h.status, "?")
            lines.append(f"- [{status_icon}] {h.id}: {h.claim}")
            lines.append(f"  status={h.status.value}, confidence={h.confidence:.2f}, "
                        f"validate_budget={h.validate_used}/{h.validate_budget}")
            for ev in h.evidence[-2:]:  # last 2 evidence entries
                lines.append(f"  evidence: {ev[:150]}")
        lines.append("")

        # Structural claims
        if self.structural_claims:
            lines.append("## Structural Claims")
            for c in self.structural_claims:
                lines.append(f"- {c}")
            lines.append("")

        # Open questions
        if self.open_questions:
            lines.append("## Open Questions")
            for q in self.open_questions[-3:]:  # cap at 3
                lines.append(f"- {q}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize for saving."""
        return {
            "hypotheses": {
                hid: {
                    "id": h.id, "claim": h.claim, "status": h.status.value,
                    "confidence": h.confidence, "validate_budget": h.validate_budget,
                    "validate_used": h.validate_used, "evidence": h.evidence,
                }
                for hid, h in self.hypotheses.items()
            },
            "structural_claims": self.structural_claims,
            "open_questions": self.open_questions,
        }
