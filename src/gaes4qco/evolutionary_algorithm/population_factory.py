from typing import List, Dict, Any

from quantum_circuit.interfaces import ICircuitFactory
from .population import Population


class PopulationFactory:
    """
    Responsável por criar instâncias de Population.
    """
    def __init__(self, circuit_factory: ICircuitFactory):
        self._circuit_factory = circuit_factory

    def create(
        self,
        population_size: int,
        num_qubits: int,
        max_depth: int,
        min_depth: int,
        use_evolutionary_strategy: bool
    ) -> Population:
        """
        Cria uma população com 'population_size' circuitos aleatórios.
        """
        if population_size == 0:
            raise ValueError("Population size cannot be zero")

        individuals = [
            self._circuit_factory.create_random_circuit(
                num_qubits=num_qubits,
                max_depth=max_depth,
                min_depth=min_depth,
                use_evolutionary_strategy=use_evolutionary_strategy
            ) for _ in range(population_size)
        ]

        return Population(individuals)

    def create_from_list_dict(self, circuits_dicts: List[Dict[str, Any]]) -> Population:
        """
        Cria uma população a partir de uma lista de dicionários (de um checkpoint).
        """
        individuals = [
            self._circuit_factory.create_from_dict(circuit_dict)
            for circuit_dict in circuits_dicts
        ]
        return Population(individuals)
