import random
from itertools import combinations
from typing import List, Iterator, Optional

from dependency_injector.wiring import Provide, inject

from analysis.interfaces import IDistanceMetric
from quantum_circuit.circuit import Circuit


class Population:
    """
    Encapsula uma coleção de indivíduos (Circuitos).
    """

    @inject
    def __init__(
        self, individuals: Optional[List[Circuit]] = None,
        distance_metric: IDistanceMetric = Provide["analysis.distance_metric"]
    ):
        self._individuals = individuals if individuals is not None else []
        self._distance_metric = distance_metric

    def add_individual(self, individual: Circuit):
        self._individuals.append(individual)

    def get_fittest(self) -> Optional[Circuit]:
        evaluated_individuals = [ind for ind in self._individuals if ind.fitness is not None]
        if not evaluated_individuals:
            return None
        return max(evaluated_individuals, key=lambda ind: ind.fitness)

    def get_individuals(self) -> List[Circuit]:
        return list(self._individuals)

    def calculate_structural_diversity(self, max_samples: int = 50) -> float:
        """
        Estima a diversidade estrutural média usando a métrica de distância injetada.
        """
        num_individuals = len(self._individuals)
        if num_individuals < 2:
            return 0.0

        # 1. Pega uma amostra representativa da população (sem repetição)
        sample_size = min(max_samples, num_individuals)
        sampled_individuals = random.sample(self._individuals, sample_size)

        total_distance = 0.0

        # 2. Gera todos os pares únicos possíveis DENTRO da amostra
        # Se max_samples = 50, isso gera (50 * 49) / 2 = 1225 pares.
        # Isso é computacionalmente leve e estatisticamente muito mais preciso.
        pairs = list(combinations(sampled_individuals, 2))
        num_pairs = len(pairs)

        if num_pairs == 0:
            return 0.0

        # 3. Calcula a distância para cada par
        for ind1, ind2 in pairs:
            total_distance += self._distance_metric.calculate(ind1, ind2)

        # 4. Retorna a média correta
        return total_distance / num_pairs

    def __len__(self) -> int:
        return len(self._individuals)

    def __iter__(self) -> Iterator[Circuit]:
        return iter(self.get_individuals())
