import random
from typing import List, Iterator

from analysis.distance_metrics import LevenshteinCircuitDistance
from quantum_circuit.circuit import Circuit


class Population:
    """
    Encapsula uma coleção de indivíduos (Circuitos).
    """
    def __init__(self, individuals: List[Circuit] = None):
        self._individuals = individuals if individuals is not None else []
        self._distance_metric = LevenshteinCircuitDistance()

    def add_individual(self, individual: Circuit):
        self._individuals.append(individual)

    def get_fittest(self) -> Circuit:
        evaluated_individuals = [ind for ind in self._individuals if ind.fitness is not None]
        if not evaluated_individuals:
            return None
        return max(evaluated_individuals, key=lambda ind: ind.fitness)

    def get_individuals(self) -> List[Circuit]:
        return list(self._individuals)

    def calculate_structural_diversity(self, sample_size: int = 100) -> float:
        """
        Estima a diversidade estrutural média usando a distância de Levenshtein.
        """
        num_individuals = len(self._individuals)
        if num_individuals < 2:
            return 0.0

        total_distance = 0.0
        effective_sample_size = min(sample_size, num_individuals)

        for _ in range(effective_sample_size):
            ind1, ind2 = random.sample(self._individuals, 2)
            total_distance += self._distance_metric.calculate(ind1, ind2)

        if effective_sample_size == 0:
            return 0.0

        return total_distance / effective_sample_size

    def __len__(self) -> int:
        return len(self._individuals)

    def __iter__(self) -> Iterator[Circuit]:
        return iter(self.get_individuals())