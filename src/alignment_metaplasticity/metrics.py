from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class AlignmentMetrics:
    authorized_acceptance: float
    unauthorized_resistance: float
    neutral_accuracy: float
    selective_corrigibility: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def harmonic_mean(a: float, b: float, eps: float = 1e-12) -> float:
    return float(2.0 * a * b / (a + b + eps))
