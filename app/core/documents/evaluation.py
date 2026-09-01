from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OCRModelCandidate:
    """Candidate OCR stack for BUILD evaluation.

    This keeps the implementation replaceable while assigning each model a clear
    production posture: baseline, candidate, or deferred.
    """

    name: str
    family: str
    openvino_supported: bool = False
    document_oriented: bool = False
    multilingual_support: str = "unknown"
    licensing: str = "unknown"
    status: str = "candidate"
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OCRCandidateEvaluation:
    candidate: OCRModelCandidate
    score: float
    reason: str


class OCRModelEvaluator:
    """Simple scoring function for model selection before implementation.

    This intentionally does not embed product- or vendor-specific logic beyond a
    weighted score used to rank candidates. It keeps the model evaluation layer
    separate from the runtime boundary and the OCR provider contract.
    """

    def evaluate_candidates(
        self,
        candidates: list[OCRModelCandidate],
        metrics: dict[str, dict[str, float]],
    ) -> list[OCRCandidateEvaluation]:
        evaluated: list[OCRCandidateEvaluation] = []

        for candidate in candidates:
            data = metrics.get(candidate.name, {})
            accuracy = float(data.get("accuracy", 0.0))
            latency = float(data.get("latency", 0.0))
            memory = float(data.get("memory", 0.0))
            languages = float(data.get("languages", 0.0))

            if candidate.openvino_supported:
                openvino_bonus = 4.0
            else:
                openvino_bonus = 0.0

            if candidate.document_oriented:
                document_bonus = 8.0
            else:
                document_bonus = 0.0

            if candidate.multilingual_support in {"broad", "very_broad"}:
                multilingual_bonus = 8.0
            elif candidate.multilingual_support == "limited":
                multilingual_bonus = 2.0
            else:
                multilingual_bonus = 0.0

            if candidate.licensing == "apache-2.0":
                license_bonus = 4.0
            elif candidate.licensing == "non-commercial":
                license_bonus = -8.0
            else:
                license_bonus = 0.0

            score = (
                0.8 * accuracy
                + 0.1 * latency
                + 0.05 * memory
                + 0.08 * languages
                + openvino_bonus
                + document_bonus
                + multilingual_bonus
                + license_bonus
            )

            evaluated.append(
                OCRCandidateEvaluation(
                    candidate=candidate,
                    score=score,
                    reason=(
                        "qualified for BUILD OCR evaluation; ranked by accuracy, "
                        "latency, memory, multilingual coverage, and deployment fit"
                    ),
                )
            )

        return sorted(evaluated, key=lambda item: item.score, reverse=True)

    def select_primary_candidate(
        self,
        candidates: list[OCRModelCandidate],
        metrics: dict[str, dict[str, float]],
    ) -> OCRModelCandidate:
        ranked = self.evaluate_candidates(candidates, metrics)
        return ranked[0].candidate
