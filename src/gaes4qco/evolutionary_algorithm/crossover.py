import random
from itertools import chain
from typing import Tuple, List

import numpy as np

from quantum_circuit.gate import Gate
from quantum_circuit.gate_factory import GateFactory
from .interfaces import IPopulationCrossover, ICrossoverStrategy
from .population import Population
from quantum_circuit.circuit import Circuit, Column


class PopulationCrossover(IPopulationCrossover):
    def __init__(self, crossover_strategy: ICrossoverStrategy, crossover_rate: float = 0.8):
        self.crossover_strategy = crossover_strategy
        self.crossover_rate = crossover_rate

    def run(self, parent_population: Population) -> Population:
        # 1. Extrai a lista de indivíduos do objeto Population.
        parents_local = parent_population.get_individuals()
        offspring = []
        indices = random.sample(range(len(parents_local)), len(parents_local))
        for i in range(0, len(indices), 2):
            if i + 1 >= len(parents_local):
                offspring.append(parents_local[indices[i]])
                continue
            parent1, parent2 = parents_local[indices[i]], parents_local[indices[i + 1]]
            if random.random() < self.crossover_rate:
                child1, child2 = self.crossover_strategy.crossover(parent1, parent2)
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1, parent2])

        return Population(offspring)


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
    """
    2D crossover that splits parents by a column and a qubit.
    Reuses existing gates only, does not create new genes.
    """
    def __init__(self, gate_factory: GateFactory):
        self._gate_factory = gate_factory

    def crossover(self, parent_1: Circuit, parent_2: Circuit) -> Tuple[Circuit, Circuit]:
        num_qubits = max(parent_1.count_qubits, parent_2.count_qubits)
        min_depth = min(parent_1.depth, parent_2.depth)

        split_col = random.randint(0, min_depth - 1)
        split_qubit = random.randint(0, num_qubits - 1)

        child1 = self._build_child(parent_1, parent_2, split_col, split_qubit, num_qubits, min_depth)
        child2 = self._build_child(parent_2, parent_1, split_col, split_qubit, num_qubits, min_depth)

        if parent_1.depth > min_depth:
            child1.columns.extend(col.copy() for col in parent_1.columns[min_depth:])
        if parent_2.depth > min_depth:
            child2.columns.extend(col.copy() for col in parent_2.columns[min_depth:])

        return child1, child2

    def _build_child(
        self, p1: Circuit, p2: Circuit, split_c: int, split_q: int, num_qubits: int, depth: int
    ) -> Circuit:
        child_cols = []

        for i in range(depth):
            new_col_gates: List[Gate] = []
            if i < split_c:
                if i < p1.depth:
                    new_col_gates.extend(g.copy() for g in p1.columns[i].get_gates())
            else:
                if i < p1.depth:
                    new_col_gates.extend(g.copy() for g in p1.columns[i].get_gates() if all(q <= split_q for q in g.qubits))
                if i < p2.depth:
                    new_col_gates.extend(g.copy() for g in p2.columns[i].get_gates() if all(q > split_q for q in g.qubits))
            free_qubits = set(range(num_qubits))
            for g in new_col_gates:
                free_qubits -= set(g.qubits)
            for q in free_qubits:
                new_col_gates.append(self._gate_factory.build_identity_gate(q))
            child_cols.append(Column(new_col_gates))

        return Circuit(num_qubits, child_cols)


class SinglePointCrossover(ICrossoverStrategy):
    """
    Crossover de Ponto Único.
    Sorteia uma única coluna para ser o ponto de troca de material genético.
    """

    def crossover(self, parent_1: Circuit, parent_2: Circuit) -> Tuple[Circuit, Circuit]:
        min_depth = min(parent_1.depth, parent_2.depth)
        if min_depth == 1:
            return parent_1.copy(), parent_2.copy()

        crossover_point = random.randint(1, min_depth - 1)
        child1_cols = [col.copy() for col in chain(parent_1.columns[:crossover_point], parent_2.columns[crossover_point:])]
        child2_cols = [col.copy() for col in chain(parent_2.columns[:crossover_point], parent_1.columns[crossover_point:])]

        num_qubits = max(parent_1.count_qubits, parent_2.count_qubits)
        child1 = Circuit(num_qubits, child1_cols)
        child2 = Circuit(num_qubits, child2_cols)
        return child1, child2
