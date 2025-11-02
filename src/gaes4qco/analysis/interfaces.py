from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from qiskit.quantum_info import Statevector

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from experiment.config import ExperimentConfig

from quantum_circuit.circuit import Circuit
from .data_models import ResultData


class IDataLoader(ABC):
    """Interface para classes que carregam dados de resultado."""
    @abstractmethod
    def load(self, filepath: str) -> ResultData:
        """Carrega os dados de um arquivo e retorna um objeto ResultData."""
        pass


class IPlotter(ABC):
    """Interface para classes que geram gráficos a partir de dados de resultado."""
    @abstractmethod
    def plot(self, data: ResultData, output_path: str):
        """Gera e salva um gráfico a partir de um objeto ResultData."""
        pass


class IDistanceMetric(ABC):

    @staticmethod
    def calculate(ind1: Circuit, ind2: Circuit) -> float:
        """Calcula a distância entre dois indivíduos."""
        pass


class IErrorAnalyzer(ABC):
    @abstractmethod
    def calculate_error_rate(self, circuit: Circuit, target_statevector: Statevector, shots: int) -> float:
        pass


class ICircuitErrorEvaluator(ABC):
    @abstractmethod
    def evaluate_circuit(self, circuit_json_path: Union[str, Path]) -> float:
        pass


class IJsonResultConcatenator(ABC):
    @abstractmethod
    def process_single_test(self, config: "ExperimentConfig", test_filename: str) -> Path:
        pass
