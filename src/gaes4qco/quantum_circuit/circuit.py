from typing import List, Tuple, Set

from .column import Column


class Circuit:
    """
    ## Entidade principal do domínio, representa um circuito quântico.
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
        
        self._structural_key: Tuple = ()
        self._structural_set: Set[Tuple] = None # Cache para o conjunto de colunas

    @property
    def objectives(self) -> Tuple[float, ...]:
        """Retorna a tupla de objetivos para a otimização multiobjetivo."""
        if self.fidelity is None:
            return -float('inf'), -float('inf')
        return self.fidelity, -float(self.depth)

    @property
    def depth(self) -> int:
        return len(self.columns)

    def get_structural_key(self) -> Tuple:
        """
        Cria uma chave única e imutável (tupla de tuplas) que representa a estrutura do circuito.
        """
        if not self._structural_key:
            col_tuples = []
            for col in self.columns:
                sorted_gates = sorted(col.get_gates(), key=lambda g: tuple(sorted(g.qubits)))
                gate_tuples = tuple(
                    (gate.gate_class.__name__, tuple(sorted(gate.qubits)))
                    for gate in sorted_gates
                )
                col_tuples.append(gate_tuples)
            self._structural_key = tuple(col_tuples)
        return self._structural_key

    def get_structural_set(self) -> Set[Tuple]:
        """
        Retorna uma representação em conjunto (set) da estrutura do circuito para comparações rápidas.
        Usa um cache para evitar a conversão repetida.
        """
        if self._structural_set is None:
            self._structural_set = set(self.get_structural_key())
        return self._structural_set

    def to_dict(self) -> dict:
        """Converte o objeto Circuit e seus componentes para um dicionário serializável."""
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
        """Retorna uma cópia leve do circuito."""
        new_circuit = Circuit(
            count_qubits=self.count_qubits,
            columns=[col.copy() for col in self.columns]
        )
        # Propaga os caches para a cópia
        new_circuit._structural_key = self._structural_key
        new_circuit._structural_set = self._structural_set
        return new_circuit

    def get_cx_count(self) -> int:
        """Conta o número de portas CX (CNOT) presentes no circuito."""
        return sum(
            1 for col in self.columns 
            for gate in col.get_gates() 
            if gate.gate_class.__name__ == 'CXGate'
        )