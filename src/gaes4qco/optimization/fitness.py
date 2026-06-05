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

    def __init__(self, target_statevector: Statevector, circuit_adapter: IQuantumCircuitAdapter,
                 max_cache_size: int = 100000):
        self._target_sv = target_statevector
        self._adapter = circuit_adapter

        # O Cache: Mapeia Chave Estrutural -> Fidelidade
        self._cache: Dict[Tuple, float] = {}
        self._max_cache_size = max_cache_size
        self.cache_hits = 0

    def evaluate(self, circuit: Circuit) -> Tuple[float, float]:
        # 1. Se o próprio objeto já tem a nota, retorna imediato (Ex: sobreviveu da geração passada)
        if circuit.base_fitness is not None and circuit.fidelity is not None:
            return circuit.base_fitness, circuit.fidelity

        # 2. Gera a chave estrutural (discretiza ângulos contínuos)
        circuit_key = circuit.get_structural_key()

        if circuit_key in self._cache:
            self.cache_hits += 1
            fidelity = self._cache[circuit_key]
        else:
            # 4. Caso Inédito: Chama o gargalo (Qiskit)
            qiskit_circuit = self._adapter.from_domain(circuit)
            solution_sv = Statevector.from_instruction(qiskit_circuit)
            fidelity = state_fidelity(solution_sv, self._target_sv)

            if len(self._cache) >= self._max_cache_size:
                logging.warning("Cache de avaliação atingiu o limite. Esvaziando para proteger memória.")
                self._cache.clear()

            self._cache[circuit_key] = fidelity

        return max(0.0, fidelity), fidelity


class WeightedFidelityFitnessEvaluator(IFitnessEvaluator):
    """
    Calcula o fitness com penalidade de profundidade,
    aproveitando o Cache Global Estrutural para a Fidelidade.
    """

    def __init__(self, target_statevector: Statevector, circuit_adapter: IQuantumCircuitAdapter, target_depth: int,
                 max_cache_size: int = 100000):
        self._target_sv = target_statevector
        self._adapter = circuit_adapter
        self._target_depth = target_depth

        # O Cache
        self._cache: Dict[Tuple, float] = {}
        self._max_cache_size = max_cache_size
        self.cache_hits = 0

    def evaluate(self, circuit: Circuit) -> Tuple[float, float]:
        if circuit.fidelity is not None:
            fidelity = circuit.fidelity
        else:
            circuit_key = circuit.get_structural_key(round_decimals=2)
            if circuit_key in self._cache:
                self.cache_hits += 1
                fidelity = self._cache[circuit_key]
            else:
                qiskit_circuit = self._adapter.from_domain(circuit)
                solution_sv = Statevector.from_instruction(qiskit_circuit)
                fidelity = state_fidelity(solution_sv, self._target_sv)

                # Controle de memória
                if len(self._cache) >= self._max_cache_size:
                    logging.warning("Cache de avaliação atingiu o limite. Esvaziando para proteger memória.")
                    self._cache.clear()

                self._cache[circuit_key] = fidelity

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
