import logging
from typing import Tuple, Dict

from qiskit.quantum_info import Statevector, state_fidelity
from .interfaces import IFitnessEvaluator
from quantum_circuit.circuit import Circuit
from quantum_circuit.interfaces import IQuantumCircuitAdapter


class FidelityFitnessEvaluator(IFitnessEvaluator):
    """
    Calcula o fitness com base na fidelidade do estado quântico,
    utilizando um Cache Global Estrutural para evitar reavaliações no Qiskit.
    """

    def __init__(self, target_statevector: Statevector, circuit_adapter: IQuantumCircuitAdapter):
        self._target_sv = target_statevector
        self._adapter = circuit_adapter

    def evaluate(self, circuit: Circuit) -> Tuple[float, float]:
        if circuit.base_fitness is not None and circuit.fidelity is not None:
            return circuit.base_fitness, circuit.fidelity
        qiskit_circuit = self._adapter.from_domain(circuit)
        solution_sv = Statevector.from_instruction(qiskit_circuit)
        fidelity = state_fidelity(solution_sv, self._target_sv)

        return max(0.0, fidelity), fidelity


class WeightedFidelityFitnessEvaluator(IFitnessEvaluator):
    """
    Calcula o fitness com penalidade de profundidade,
    aproveitando o Cache Global Estrutural para a Fidelidade.
    """

    def __init__(self, target_statevector: Statevector, circuit_adapter: IQuantumCircuitAdapter, target_depth: int):
        self._target_sv = target_statevector
        self._adapter = circuit_adapter
        self._target_depth = target_depth

    def evaluate(self, circuit: Circuit) -> Tuple[float, float]:
        if circuit.fidelity is not None:
            fidelity = circuit.fidelity
        else:

            qiskit_circuit = self._adapter.from_domain(circuit)
            solution_sv = Statevector.from_instruction(qiskit_circuit)
            fidelity = state_fidelity(solution_sv, self._target_sv)

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


class CNOTPenaltyFitnessEvaluator(IFitnessEvaluator):
    """
    Calcula o fitness com penalidade focada no hardware físico (NISQ).
    Em vez de punir a profundidade total, pune a quantidade de portas de 2-qubits (CXGates),
    pois são elas que dominam o erro físico e forçam o roteamento de SWAPs.
    """

    def __init__(self, target_statevector: Statevector, circuit_adapter: IQuantumCircuitAdapter, max_allowed_cx: int):
        self._target_sv = target_statevector
        self._adapter = circuit_adapter
        self._max_cx = max_allowed_cx  # Referência de CNOTs máximos aceitáveis

    def evaluate(self, circuit: Circuit) -> Tuple[float, float]:
        # 1. Base Fidelity (Rápido e determinístico)
        if circuit.fidelity is not None:
            fidelity = circuit.fidelity
        else:
            qiskit_circuit = self._adapter.from_domain(circuit)
            solution_sv = Statevector.from_instruction(qiskit_circuit)
            fidelity = state_fidelity(solution_sv, self._target_sv)

        # 2. Conta os CNOTs em vez da profundidade
        cx_count = circuit.get_cx_count()
        cx_ratio = cx_count / self._max_cx if self._max_cx > 0 else cx_count

        # 3. Penalidade Dinâmica
        # Só pune o circuito quando ele começa a ficar matematicamente bom (> 0.90)
        penalty_factor = fidelity ** 100

        # Exemplo: Se o circuito estourar o limite de CX, perde 15% do fitness
        cx_penalty = 1.0 - (0.15 * cx_ratio * penalty_factor)
        cx_penalty = max(0.0, cx_penalty)

        final_fitness = fidelity * cx_penalty
        return max(0.0, final_fitness), fidelity
