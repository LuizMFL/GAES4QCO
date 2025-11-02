import numpy as np

from qiskit.quantum_info import Statevector, state_fidelity

from analysis.interfaces import IErrorAnalyzer
from quantum_circuit.circuit import Circuit
from quantum_circuit.interfaces import IQuantumExecutor


class ErrorAnalyzer(IErrorAnalyzer):
    """Calcula a taxa de erro de um circuito comparando sua execução
    com o resultado ideal, seja em statevector ou counts ruidosos."""

    def __init__(self, executor: IQuantumExecutor):
        self._executor = executor

    def calculate_error_rate(self, circuit: Circuit, target_statevector: Statevector, shots: int) -> float:
        """
        Executa o circuito e calcula a taxa de erro.
        Funciona tanto para backend determinístico (statevector) quanto ruidoso (counts).
        """
        # Estado com maior probabilidade no target
        ideal_probs = target_statevector.probabilities_dict()
        # Executa o circuito
        result = self._executor.execute(circuit, shots, measure=True)
        if isinstance(result, dict):
            # Distribuição observada (normalizada)
            measured_probs = {k: v / shots for k, v in result.items()}

            # Garante que todos os estados do ideal estão presentes em measured_probs
            all_states = set(ideal_probs.keys()) | set(measured_probs.keys())
            p = np.array([ideal_probs.get(k, 0.0) for k in all_states])
            q = np.array([measured_probs.get(k, 0.0) for k in all_states])

            # ---- TAXA DE ERRO GLOBAL: Total Variation Distance ----
            total_variation = 0.5 * np.sum(np.abs(p - q))
            error_rate = total_variation

            print(f"[Simulador] TV distance = {total_variation:.4f} → Erro global = {error_rate:.2%}")
            return float(error_rate)
        elif isinstance(result, Statevector):
            fidelity = state_fidelity(result, target_statevector)
            error_rate = 1.0 - fidelity
            print(f"Fidelidade: {fidelity:.4f} → Erro global = {error_rate:.2%}")
            return float(error_rate)
        else:
            raise TypeError(f"Executor retornou tipo inesperado: {type(result)}")