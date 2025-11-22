import json
from pathlib import Path
from typing import Union

from qiskit.quantum_info import Statevector

from .interfaces import IErrorAnalyzer, ICircuitErrorEvaluator
from quantum_circuit.interfaces import ICircuitFactory, IQuantumCircuitAdapter


class CircuitErrorEvaluator(ICircuitErrorEvaluator):
    """
    ### CircuitErrorEvaluator
    Avalia e compara taxas de erro de circuitos quânticos
    em relação a um circuito alvo (Target Circuit).

    - Carrega e armazena automaticamente o circuito alvo e seu `Statevector`.
    - Pode avaliar circuitos otimizados reutilizando o `Statevector` alvo.
    - Segue princípios SOLID e facilita testes unitários.
    """

    def __init__(
        self,
        target_circuit_path: Union[str, Path],
        circuit_factory: ICircuitFactory,
        qiskit_adapter: IQuantumCircuitAdapter,
        error_analyzer: IErrorAnalyzer,
        shots: int,
        verbose: bool = True,
    ):
        """
        Inicializa o avaliador e carrega o circuito alvo imediatamente.

        Args:
            target_circuit_path: Caminho do arquivo JSON do circuito alvo.
            circuit_factory: Fábrica responsável por criar Circuitos do domínio.
            qiskit_adapter: Adaptador Circuit (domínio) → Qiskit.
            error_analyzer: Componente que calcula a taxa de erro.
            shots: Número de execuções do simulador quântico.
        """
        self._target_path = Path(target_circuit_path)
        self._circuit_factory = circuit_factory
        self._adapter = qiskit_adapter
        self._error_analyzer = error_analyzer
        self._shots = shots
        self._verbose = verbose

        # Carregamento imediato do circuito alvo
        self._target_statevector = self._load_target_statevector(self._target_path)

    def _load_target_statevector(self, path: Path):
        """Carrega e converte o circuito alvo, retornando (domínio, statevector)."""
        if not path.exists():
            raise FileNotFoundError(f"O circuito alvo não foi encontrado em: {path}")

        with open(path, "r") as f:
            circuit_data = json.load(f)

        circuit_domain = self._circuit_factory.create_from_dict(circuit_data)
        qiskit_circuit = self._adapter.from_domain(circuit_domain)
        statevector = Statevector.from_instruction(qiskit_circuit)

        if self._verbose: print(f"[INFO] Circuito alvo carregado com sucesso: {path.name}")
        return statevector

    def evaluate_circuit(self, circuit_json_path: Union[str, Path]) -> float:
        """
        Avalia a taxa de erro de um circuito comparando com o `Statevector` alvo.

        Args:
            circuit_json_path: Caminho do arquivo JSON com o circuito a testar.
        Returns:
            float: Taxa de erro calculada.
        """
        circuit_json_path = Path(circuit_json_path)
        if not circuit_json_path.exists():
            raise FileNotFoundError(f"Arquivo de circuito não encontrado: {circuit_json_path}")

        with open(circuit_json_path, "r") as f:
            circuit_data = json.load(f)

        circuit_domain = self._circuit_factory.create_from_dict(circuit_data)
        error_rate = self._error_analyzer.calculate_error_rate(
            circuit=circuit_domain,
            target_statevector=self._target_statevector,
            shots=self._shots,
            verbose=self._verbose,
        )

        return error_rate

