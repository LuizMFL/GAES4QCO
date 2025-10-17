from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class ResultData:
    """
    Contém os dados de resultado de uma única execução do algoritmo.
    """
    fitness_per_generation: List[List[float]]
    structural_diversity_per_generation: List[float]
    fidelity_per_generation: List[List[float]]
    depth_per_generation: List[List[float]]

    @property
    def generation_count(self) -> int:
        """Retorna o número de gerações registradas."""
        return len(self.average_fitness_per_generation)

    @property
    def average_fitness_per_generation(self) -> List[float]:
        """Média da fidelidade dos indivíduos por geração."""
        return [
            float(np.mean(gen)) if gen else 0.0
            for gen in self.fitness_per_generation
        ]

    @property
    def std_dev_fitness_per_generation(self) -> List[float]:
        """Desvio padrão da fidelidade entre os indivíduos de cada geração."""
        return [
            float(np.std(gen, ddof=0)) if gen else 0.0
            for gen in self.fitness_per_generation
        ]

    @property
    def best_fitness_per_generation(self) -> List[float]:
        """Melhor fidelidade (máximo) de cada geração."""
        return [gen[0] if gen else 0.0 for gen in self.fitness_per_generation]


    @property
    def average_fidelity_per_generation(self) -> List[float]:
        """Média da fidelidade dos indivíduos por geração."""
        return [
            float(np.mean(gen)) if gen else 0.0
            for gen in self.fidelity_per_generation
        ]

    @property
    def std_dev_fidelity_per_generation(self) -> List[float]:
        """Desvio padrão da fidelidade entre os indivíduos de cada geração."""
        return [
            float(np.std(gen, ddof=0)) if gen else 0.0
            for gen in self.fidelity_per_generation
        ]

    @property
    def best_fidelity_per_generation(self) -> List[float]:
        """Melhor fidelidade (máximo) de cada geração."""
        return [gen[0] if gen else 0.0 for gen in self.fidelity_per_generation]


    @property
    def average_depth_per_generation(self) -> List[float]:
        """Profundidade média das estruturas por geração (se disponível)."""
        if not self.depth_per_generation:
            return []
        return [
            float(np.mean(gen)) if gen else 0.0
            for gen in self.depth_per_generation
        ]

    @property
    def std_dev_depth_per_generation(self) -> List[float]:
        """Desvio padrão da fidelidade entre os indivíduos de cada geração."""
        return [
            float(np.std(gen, ddof=0)) if gen else 0.0
            for gen in self.depth_per_generation
        ]

    @property
    def max_depth_per_generation(self) -> List[float]:
        """Profundidade máxima observada por geração (se disponível)."""
        if not self.depth_per_generation:
            return []
        return [max(gen) if gen else 0.0 for gen in self.depth_per_generation]