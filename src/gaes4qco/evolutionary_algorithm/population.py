from itertools import combinations
import random
from typing import List, Iterator

from analysis.distance_metrics import StructuralJaccardDistance
from quantum_circuit.circuit import Circuit


class Population:
    """
    Encapsula uma coleção de indivíduos (Circuitos) e fornece
    operações úteis sobre o conjunto.
    """
    def __init__(self, individuals: List[Circuit] = None):
        self._individuals = individuals if individuals is not None else []
        self._distance_metric = StructuralJaccardDistance()

    def add_individual(self, individual: Circuit):
        """Adiciona um indivíduo à população."""
        self._individuals.append(individual)

    def get_fittest(self) -> Circuit:
        """Encontra e retorna o indivíduo com o maior fitness."""
        if not self._individuals:
            return None
        
        evaluated_individuals = [ind for ind in self._individuals if ind.fitness is not None]
        if not evaluated_individuals:
            return None
            
        return max(evaluated_individuals, key=lambda ind: ind.fitness)

    def get_individuals(self) -> List[Circuit]:
        """Retorna a lista de todos os indivíduos."""
        return list(self._individuals)

    @property
    def average_fitness(self) -> float:
        """Calcula e retorna a média do fitness da população."""
        evaluated_fitness = [ind.fitness for ind in self._individuals if ind.fitness is not None]
        if not evaluated_fitness:
            return 0.0
        return sum(evaluated_fitness) / len(evaluated_fitness)

    def calculate_structural_diversity(self, sample_size: int = 100) -> float:
        """
        Estima a diversidade estrutural média da população usando amostragem.
        Em vez de comparar todos os pares (O(n^2)), seleciona uma amostra aleatória
        de pares para um cálculo muito mais rápido (O(sample_size)).
        """
        num_individuals = len(self._individuals)
        if num_individuals < 2:
            return 0.0

        total_distance = 0.0
        
        # Garante que o tamanho da amostra não seja maior que o número de indivíduos
        effective_sample_size = min(sample_size, num_individuals)

        # Seleciona 'effective_sample_size' indivíduos aleatórios para formar os pares
        # Isso é mais eficiente do que gerar todas as combinações e depois amostrar
        for _ in range(effective_sample_size):
            ind1, ind2 = random.sample(self._individuals, 2)
            total_distance += self._distance_metric.calculate(ind1, ind2)

        if effective_sample_size == 0:
            return 0.0

        return total_distance / effective_sample_size

    def without_duplicates(self) -> "Population":
        """
        Retorna uma nova população sem indivíduos estruturalmente duplicados.
        Usa a chave estrutural canônica para identificar duplicatas.
        """
        seen_keys = set()
        unique_individuals = []
        for individual in self._individuals:
            key = individual.get_structural_key()
            if key not in seen_keys:
                seen_keys.add(key)
                unique_individuals.append(individual)
        return Population(unique_individuals)

    def __len__(self) -> int:
        """Permite o uso de len(population_object)."""
        return len(self._individuals)

    def __iter__(self) -> Iterator[Circuit]:
        """Permite iterar sobre os indivíduos: for circuit in population_object:"""
        return iter(self.get_individuals())