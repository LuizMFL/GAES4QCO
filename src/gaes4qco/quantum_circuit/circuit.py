from typing import List, Tuple, Set, Optional

from .column import Column


class Circuit:
    """
    Representa um circuito quântico. É um objeto de dados puro, sem lógica de cache.
    """

    def __init__(
            self,
            count_qubits: int,
            columns: List[Column],
            fitness: Optional[float] = None,
            base_fitness: Optional[float] = None,
            fidelity: Optional[float] = None
    ):
        self.count_qubits = count_qubits
        self.columns = columns
        self.fitness = fitness
        self.base_fitness = fitness
        self.base_fitness = base_fitness
        self.fidelity = fidelity
        self.rank: int = -1
        self.crowding_distance: float = 0.0

    @property
    def objectives(self) -> Tuple[float, ...]:
        if self.base_fitness is None:
            return -float('inf'), -float('inf')

        return self.base_fitness, -float(self.depth)

    @property
    def depth(self) -> int:
        return len(self.columns)

    def get_structural_key(self, round_decimals: int = 5) -> Tuple:
        """
        Calcula e retorna uma chave única e imutável que representa a estrutura do circuito.
        Mantém a direcionalidade de gates multi-qubits intacta e discretiza parâmetros
        contínuos para diferenciar variações angulares significativas na distância de Levenshtein.
        """
        col_tuples = []
        for col in self.columns:
            # 1. Ordena os gates da coluna pela sua tupla ORIGINAL de qubits
            sorted_gates = sorted(col.get_gates(), key=lambda g: tuple(g.qubits))

            # 2. Cria a assinatura do gate preservando a ordem Control/Target e os parâmetros
            gate_tuples = []
            for gate in sorted_gates:
                # Arredonda os parâmetros contínuos para a precisão desejada (discretização)
                rounded_params = tuple(round(p, round_decimals) for p in gate.parameters) if gate.parameters else ()

                gate_sig = (gate.gate_class.__name__, tuple(gate.qubits), rounded_params)
                gate_tuples.append(gate_sig)

            col_tuples.append(tuple(gate_tuples))

        return tuple(col_tuples)

    def to_dict(self) -> dict:
        return {
            "count_qubits": self.count_qubits,
            "depth": self.depth,
            "fitness": self.fitness,
            "base_fitness": self.base_fitness,
            "fidelity": self.fidelity,
            "nsga2_rank": self.rank,
            "nsga2_crowding_distance": self.crowding_distance,
            "columns": [col.to_dict() for col in self.columns]
        }

    def copy(self) -> "Circuit":
        return Circuit(
            count_qubits=self.count_qubits,
            columns=[col.copy() for col in self.columns],
            fitness=self.fitness,
            base_fitness=self.base_fitness,
            fidelity=self.fidelity
        )

    def get_cx_count(self) -> int:
        return sum(
            1 for col in self.columns 
            for gate in col.get_gates() 
            if gate.gate_class.__name__ == 'CXGate'
        )
