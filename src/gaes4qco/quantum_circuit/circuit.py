from typing import List, Tuple, Set

from .column import Column


class Circuit:
    """
    Representa um circuito quântico. É um objeto de dados puro, sem lógica de cache.
    """

    def __init__(
            self,
            count_qubits: int,
            columns: List[Column],
            fitness: float = None,
            fidelity: float = None
    ):
        self.count_qubits = count_qubits
        self.columns = columns
        self.fitness = fitness
        self.fidelity = fidelity
        self.rank: int = -1
        self.crowding_distance: float = 0.0

    @property
    def objectives(self) -> Tuple[float, ...]:
        if self.fidelity is None:
            return -float('inf'), -float('inf')
        return self.fidelity, -float(self.depth)

    @property
    def depth(self) -> int:
        return len(self.columns)

    def get_structural_key(self) -> Tuple:
        """
        Calcula e retorna uma chave única e imutável que representa a estrutura do circuito.
        Esta operação é computacionalmente intensiva e não deve ser chamada em loops apertados.
        """
        col_tuples = []
        for col in self.columns:
            sorted_gates = sorted(col.get_gates(), key=lambda g: tuple(sorted(g.qubits)))
            gate_tuples = tuple(
                (gate.gate_class.__name__, tuple(sorted(gate.qubits)))
                for gate in sorted_gates
            )
            col_tuples.append(gate_tuples)
        return tuple(col_tuples)

    def to_dict(self) -> dict:
        return {
            "count_qubits": self.count_qubits,
            "depth": self.depth,
            "fitness": self.fitness,
            "fidelity": self.fidelity,
            "nsga2_rank": self.rank,
            "nsga2_crowding_distance": self.crowding_distance,
            "columns": [col.to_dict() for col in self.columns]
        }

    def copy(self) -> "Circuit":
        return Circuit(
            count_qubits=self.count_qubits,
            columns=[col.copy() for col in self.columns]
        )

    def get_cx_count(self) -> int:
        return sum(
            1 for col in self.columns 
            for gate in col.get_gates() 
            if gate.gate_class.__name__ == 'CXGate'
        )