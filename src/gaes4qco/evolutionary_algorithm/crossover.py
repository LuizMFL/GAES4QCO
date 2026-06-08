import random
from itertools import chain
from typing import Tuple, List

import numpy as np

from .interfaces import IPopulationCrossover, ICrossoverStrategy
from .population import Population
from quantum_circuit.circuit import Circuit, Column
from quantum_circuit.gate_factory import GateFactory


class PopulationCrossover(IPopulationCrossover):
    """
    Responsável por criar a população de filhos (offspring).
    Aplica crossover ou, se não aplicar, força a mutação para garantir a variação.
    """
    def __init__(
        self,
        crossover_strategy: ICrossoverStrategy,
        crossover_rate: float
    ):
        self.crossover_strategy = crossover_strategy
        self.crossover_rate = crossover_rate

    def run(self, parent_population: Population):
        parents_local = parent_population.get_individuals()
        offspring = []
        parents_final = []
        indices = random.sample(range(len(parents_local)), len(parents_local))

        for i in range(0, len(indices), 2):
            if i + 1 >= len(parents_local):
                parents_final.append(parents_local[indices[i]])
                continue
                
            parent1, parent2 = parents_local[indices[i]], parents_local[indices[i + 1]]
            
            if random.random() < self.crossover_rate:
                child1, child2 = self.crossover_strategy.crossover(parent1, parent2)
                offspring.extend([child1, child2])
            else:
                parents_final.extend([parent1, parent2])

        return Population(offspring), Population(parents_final)


class MultiPointCrossover(ICrossoverStrategy):
    def crossover(self, parent_1: Circuit, parent_2: Circuit) -> Tuple[Circuit, Circuit]:
        min_depth = min(parent_1.depth, parent_2.depth)
        child1_cols, child2_cols = [], []
        crossover_points = np.random.randint(0, 2, size=min_depth, dtype=np.int8)
        for i, use_p1 in enumerate(crossover_points):
            src1, src2 = (parent_1, parent_2) if use_p1 else (parent_2, parent_1)
            child1_cols.append(src1.columns[i].copy())
            child2_cols.append(src2.columns[i].copy())

        if parent_1.depth > min_depth:
            child1_cols.extend([col.copy() for col in parent_1.columns[min_depth:]])
        if parent_2.depth > min_depth:
            child2_cols.extend([col.copy() for col in parent_2.columns[min_depth:]])

        num_qubits = parent_1.count_qubits
        return Circuit(num_qubits, child1_cols), Circuit(num_qubits, child2_cols)


class BlockwiseCrossover(ICrossoverStrategy):
    def __init__(self, gate_factory: GateFactory):
        self._gate_factory = gate_factory

    def crossover(self, parent_1: Circuit, parent_2: Circuit) -> Tuple[Circuit, Circuit]:
        num_qubits = max(parent_1.count_qubits, parent_2.count_qubits)
        max_depth = max(parent_1.depth, parent_2.depth)

        split_col = random.randint(0, max_depth)
        split_qubit = random.randint(0, num_qubits - 1)

        child1 = self._build_child(parent_1, parent_2, split_col, split_qubit, num_qubits, max_depth)
        child2 = self._build_child(parent_2, parent_1, split_col, split_qubit, num_qubits, max_depth)

        return child1, child2

    def _build_child(
        self, p1: Circuit, p2: Circuit, split_c: int, split_q: int, num_qubits: int, depth: int
    ) -> Circuit:
        child_cols = []
        for i in range(depth):
            new_col = Column()
            if i < split_c:
                source = p1
                qubit_condition = lambda q: q <= split_q
            else:
                source = p2
                qubit_condition = lambda q: q <= split_q
            
            if i < source.depth:
                for gate in source.columns[i].get_gates():
                    if all(qubit_condition(q) for q in gate.qubits):
                        new_col.add_gate(gate.copy())

            # Quadrante Inferior-Esquerdo (antes do tempo, qubits abaixo)
            if i < split_c:
                source = p2
                qubit_condition = lambda q: q > split_q
            # Quadrante Inferior-Direito (depois do tempo, qubits abaixo)
            else:
                source = p1
                qubit_condition = lambda q: q > split_q

            if i < source.depth:
                for gate in source.columns[i].get_gates():
                    if all(qubit_condition(q) for q in gate.qubits):
                        new_col.add_gate(gate.copy())
            
            # Preenche os qubits não utilizados com portas de identidade para manter a estrutura
            used_qubits = {q for g in new_col.get_gates() for q in g.qubits}
            for q in range(num_qubits):
                if q not in used_qubits:
                    new_col.add_gate(self._gate_factory.build_identity_gate(q))

            child_cols.append(new_col)

        return Circuit(num_qubits, child_cols)


class SinglePointCrossover(ICrossoverStrategy):
    """
    Crossover de Ponto Único.
    Sorteia uma única coluna para ser o ponto de troca de material genético.
    """

    def crossover(self, parent_1: Circuit, parent_2: Circuit) -> Tuple[Circuit, Circuit]:
        min_depth = min(parent_1.depth, parent_2.depth)
        if min_depth < 2:
            return parent_1.copy(), parent_2.copy()

        crossover_point = random.randint(1, min_depth - 1)

        p1_cols = parent_1.columns
        p2_cols = parent_2.columns

        # O fatiamento em Python já captura a lista até o final absoluto.
        # Child 1 assume o prefixo do P1 e a cauda (até o fim) do P2.
        child1_cols = [col.copy() for col in chain(p1_cols[:crossover_point], p2_cols[crossover_point:])]

        # Child 2 assume o prefixo do P2 e a cauda (até o fim) do P1.
        child2_cols = [col.copy() for col in chain(p2_cols[:crossover_point], p1_cols[crossover_point:])]

        num_qubits = max(parent_1.count_qubits, parent_2.count_qubits)
        return Circuit(num_qubits, child1_cols), Circuit(num_qubits, child2_cols)
