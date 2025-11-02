import random
from typing import List, Dict, Any
from .circuit import Circuit
from .column import Column
from .gate_factory import GateFactory  # <-- Importa a nova factory
from .interfaces import ICircuitFactory


class CircuitFactory(ICircuitFactory):
    """
    Responsável por criar instâncias da nossa entidade Circuit,
    delegando a criação de gates para a GateFactory.
    """

    def __init__(self, gate_factory: GateFactory):
        # A CircuitFactory agora tem uma instância da GateFactory.
        self._gate_factory = gate_factory

    def create_random_circuit(self, num_qubits: int, max_depth: int, min_depth: int, use_evolutionary_strategy: bool) -> Circuit:
        depth = random.randint(min_depth, max_depth)
        columns: List[Column] = []

        for _ in range(depth):
            qubits_free_in_column = list(range(num_qubits))
            gates_in_column = []

            # Preenche a coluna com gates até não haver mais qubits livres
            while qubits_free_in_column:
                try:
                    # Delega a criação do gate para a GateFactory
                    new_gate = self._gate_factory.build_gate(qubits_free_in_column, use_evolutionary_strategy)
                    gates_in_column.append(new_gate)

                    # Remove os qubits usados da lista de disponíveis na coluna
                    for used_qubit in new_gate.qubits:
                        qubits_free_in_column.remove(used_qubit)
                except ValueError:
                    # Ocorre se não for possível criar mais gates com os qubits restantes
                    break

            columns.append(Column(gates=gates_in_column))

        return Circuit(count_qubits=num_qubits, columns=columns)

    def create_from_dict(self, data: Dict[str, Any]) -> Circuit:
        """
        Reconstrói uma entidade Circuit a partir de um dicionário (proveniente de um JSON).
        """

        columns = [
            Column([
                self._gate_factory.create_from_dict(gate_data)
                for gate_data in column_data.get("gates")
            ])
            for column_data in data.get("columns")
        ]

        return Circuit(
            count_qubits=data.get("count_qubits"),
            columns=columns,
            fitness=data.get("fitness"),
            fidelity=data.get("fidelity")
        )
