from dataclasses import dataclass, field
from typing import List


@dataclass
class StepSize:
    """
    Encapsula o tamanho do passo (sigma) e o histórico de sucesso para a Estratégia Evolucionária.
    """
    sigma: float = 0.5
    history_len: int = 5
    history: List[int] = field(default_factory=list)

    def to_dict(self):
        return {
            "sigma": self.sigma,
            "history_len": self.history_len,
            "history": self.history
        }

    def copy(self) -> "StepSize":
        """
        Return a lightweight copy of the StepSize instance.
        """
        return StepSize(
            history_len=self.history_len,
            history=list(self.history),
            sigma=self.sigma
        )


@dataclass
class EvolutionRates:
    """Contém as taxas de crossover e mutação para uma geração."""
    crossover_rate: float
    mutation_rate: float

    def __post_init__(self):
        if not (0 <= self.crossover_rate <= 1):
            raise ValueError("Crossover rate must be in [0, 1].")
        if not (0 <= self.mutation_rate <= 1):
            raise ValueError("Mutation rate must be in [0, 1].")


class CrossoverType:
    SINGLEPOINT = "singlepoint"
    MULTIPOINT = "multipoint"
    BLOCKWISE = "blockwise"
