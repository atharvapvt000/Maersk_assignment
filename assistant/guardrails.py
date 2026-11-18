"""Code-enforced guardrails (the prompt asks; these make sure).

1. Grounding: every number in the final answer must be traceable to a tool result from the same
   question (allowing for rounding and for 0.404 <-> 40.4%), to the user's own question, or to a
   small set of domain constants. Answers with numbers must also cite a tool result id ([T1]).
2. Tool budget: hard caps per question and per session (enforced in the agent loop).
3. Off-domain: the model is instructed to answer with OFF_DOMAIN_REPLY and call no tools; the
   grounding check means an off-domain answer that invents figures is still caught.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

OFF_DOMAIN_REPLY = (
    "I can only help with questions about shipments, routes, ports, delays and delay-risk "
    "predictions in this supply-chain dataset."
)

# Numbers the assistant may state without a tool: the on-time threshold and calendar words.
DOMAIN_CONSTANTS = {24.0, 1.0, 2.0, 3.0, 4.0, 100.0}

_CITATION = re.compile(r"\[T\d+(?:\s*,\s*T\d+)*\]")
_NUMBER = re.compile(r"(?<![A-Za-z])-?\d+(?:\.\d+)?")


def _numbers(text: str) -> list[str]:
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # 1,234 -> 1234
    return _NUMBER.findall(text)


def _numeric_values(obj: Any) -> set[float]:
    """Every number appearing anywhere in a JSON-like structure (incl. inside strings/dates)."""
    out: set[float] = set()
    for token in _numbers(json.dumps(obj, default=str)):
        out.add(float(token))
    return out


def _consistent(answer_token: str, source: float) -> bool:
    decimals = len(answer_token.split(".")[1]) if "." in answer_token else 0
    value = float(answer_token)
    tolerance = 0.5 * 10**-decimals + 1e-9
    candidates = (source, source * 100, abs(source), abs(source) * 100)
    return any(abs(value - c) <= tolerance for c in candidates)


@dataclass
class GroundingVerdict:
    ok: bool
    ungrounded: list[str] = field(default_factory=list)
    missing_citation: bool = False
    cited_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "ungrounded_numbers": self.ungrounded,
            "missing_citation": self.missing_citation,
            "cited_tool_results": self.cited_ids,
        }


def check_grounding(
    answer: str, question: str, tool_outputs: list[Any], trusted: Any = None
) -> GroundingVerdict:
    """`tool_outputs`: results of this question's tool calls (figures from these need a [T#]
    citation). `trusted`: operator-supplied facts from the system prompt (no citation needed)."""
    cited = sorted(set(re.findall(r"T\d+", " ".join(_CITATION.findall(answer)))))
    body = _CITATION.sub(" ", answer)
    sources = set().union(*(_numeric_values(o) for o in tool_outputs)) if tool_outputs else set()
    allowed_free = {float(t) for t in _numbers(question)} | DOMAIN_CONSTANTS
    trusted_values = _numeric_values(trusted) if trusted else set()

    ungrounded: list[str] = []
    needs_citation = False
    for token in _numbers(body):
        value = float(token)
        if value in allowed_free or abs(value) in allowed_free:
            continue
        if any(_consistent(token, s) for s in sources):
            needs_citation = True
        elif not any(_consistent(token, s) for s in trusted_values):
            ungrounded.append(token)

    missing_citation = needs_citation and not cited
    return GroundingVerdict(
        ok=not ungrounded and not missing_citation,
        ungrounded=sorted(set(ungrounded), key=ungrounded.index),
        missing_citation=missing_citation,
        cited_ids=cited,
    )
