from typing import Tuple

from qiskit.quantum_info import Statevector, state_fidelity
from .interfaces import IFitnessEvaluator
from quantum_circuit.circuit import Circuit
from quantum_circuit.interfaces import IQuantumCircuitAdapter


class FidelityFitnessEvaluator(IFitnessEvaluator):
    """
    ## Calcula o fitness com base na fidelidade do estado quântico.
    ## Esta classe substitui a antiga função 'fidelityFitnessFunction'.
    """

    def __init__(self, target_statevector: Statevector, circuit_adapter: IQuantumCircuitAdapter):
        self._target_sv = target_statevector
        self._adapter = circuit_adapter

    def evaluate(self, circuit: Circuit) -> Tuple[float, float]:
        """
        Converte o circuito de domínio para Qiskit usando o adapter
        e então calcula a fidelidade.
        """
        # ## A conversão é delegada ao adapter, respeitando a Inversão de Dependência
        qiskit_circuit = self._adapter.from_domain(circuit)

        # ## O cálculo de fidelidade do Qiskit é realizado aqui
        solution_sv = Statevector.from_instruction(qiskit_circuit)
        fidelity = state_fidelity(solution_sv, self._target_sv)
        return max(0.0, fidelity), fidelity


class WeightedFidelityFitnessEvaluator(IFitnessEvaluator):
    """
    Calcula o fitness combinando a fidelidade com uma penalidade baseada na profundidade.
    A penalidade aumenta drasticamente quando a fidelidade se aproxima de 1.0.
    """

    def __init__(self, target_statevector: Statevector, circuit_adapter: IQuantumCircuitAdapter, target_depth: int):
        self._target_sv = target_statevector
        self._adapter = circuit_adapter
        self._target_depth = target_depth  # Profundidade do circuito alvo para normalização

    def evaluate(self, circuit: Circuit) -> Tuple[float, float]:
        # 1. Calcula a fidelidade pura
        qiskit_circuit = self._adapter.from_domain(circuit)
        solution_sv = Statevector.from_instruction(qiskit_circuit)
        fidelity = state_fidelity(solution_sv, self._target_sv)

        # 3. Calcula a penalidade de profundidade
        depth_ratio = circuit.depth / self._target_depth if self._target_depth > 0 else circuit.depth

        # A penalidade é ponderada pela fidelidade.
        # Quando a fidelidade é baixa, a penalidade é quase zero.
        # Quando a fidelidade é alta (ex: 0.99), a penalidade se torna significativa.
        penalty_factor = fidelity ** 100  # O expoente alto ativa a penalidade apenas perto de 1.0
        depth_penalty = 1.0 - (0.1 * depth_ratio * penalty_factor)  # Ex: penalidade de 10%

        # Garante que a penalidade não seja negativa
        depth_penalty = max(0.0, depth_penalty)

        # 4. O fitness final é a fidelidade ponderada pela penalidade de profundidade
        final_fitness = fidelity * depth_penalty
        return max(0.0, final_fitness), fidelity
