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
        parents_local = parent_population.get_individuals()
        offspring = []
        indices = random.sample(range(len(parents_local)), len(parents_local))
        
        for i in range(0, len(indices), 2):
            if i + 1 >= len(parents_local):
                offspring.append(parents_local[indices[i]].copy())
                continue
                
            parent1, parent2 = parents_local[indices[i]], parents_local[indices[i + 1]]
            
            if random.random() < self.crossover_rate:
                child1, child2 = self.crossover_strategy.crossover(parent1, parent2)
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1.copy(), parent2.copy()])
                
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

        # Cria os filhos trocando as "caudas"
        child1_cols = [col.copy() for col in chain(p1_cols[:crossover_point], p2_cols[crossover_point:])]
        child2_cols = [col.copy() for col in chain(p2_cols[:crossover_point], p1_cols[crossover_point:])]

        # Adiciona o restante do pai mais longo, se houver
        if len(p1_cols) > len(p2_cols):
            child2_cols.extend([col.copy() for col in p1_cols[len(p2_cols):]])
        elif len(p2_cols) > len(p1_cols):
            child1_cols.extend([col.copy() for col in p2_cols[len(p1_cols):]])

        num_qubits = max(parent_1.count_qubits, parent_2.count_qubits)
        return Circuit(num_qubits, child1_cols), Circuit(num_qubits, child2_cols)
