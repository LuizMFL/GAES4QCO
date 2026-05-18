from abc import ABC, abstractmethod
from typing import Tuple

from quantum_circuit.circuit import Circuit
from evolutionary_algorithm.population import Population


class IFitnessEvaluator(ABC):
    """Interface para qualquer classe que calcula o fitness de um circuito."""

    @abstractmethod
    def evaluate(self, circuit: Circuit) -> Tuple[float, float]:
        """Calcula e retorna o valor de fitness de um único circuito."""
        pass


class IFitnessShaper(ABC):
    """Interface para classes que ajustam/modelam o fitness de uma população inteira."""
    @abstractmethod
    def shape(self, population: Population):
        """Aplica uma transformação nos valores de fitness dos indivíduos da população."""
        pass


class IProgressObserver(ABC):
    """Interface para classes que observam e registram o progresso do algoritmo."""

    @abstractmethod
    def update(self, generation: int, population: Population, mutation_rate: float = 0.0, crossover_rate: float = 0.0):
        """Método chamado a cada geração para registrar o estado da população."""
        pass

    @abstractmethod
    def save(self):
        """Salva os dados coletados ao final da execução."""
        pass

    @abstractmethod
    def set_duration(self, duration_seconds: float, final_generation: int):
        """Define o tempo total de execução da fase."""
        pass

    @abstractmethod
    def set_summary(self, duration_seconds: float, final_generation: int, total_evaluations: int, stopping_reason: str, best_circuit: Circuit):
        pass
