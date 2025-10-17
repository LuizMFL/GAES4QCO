# optimization/observer.py

import json
import numpy as np
from .interfaces import IProgressObserver
from evolutionary_algorithm.population import Population


class JsonProgressObserver(IProgressObserver):
    """
    ## Implementa um observador que coleta estatísticas e salva em um arquivo JSON.
    """

    def __init__(self, filename: str):
        self._filename = filename
        self._data_to_save = {
            "fitness_per_generation": [],
            "fidelity_per_generation": [],
            "structural_diversity_per_generation": [],
            "depth_per_generation": [],
        }

    def update(self, generation: int, population: Population):
        """Coleta os dados de fitness da população atual."""
        individuals = population.get_individuals()
        individuals.sort(key=lambda individual: (individual.fitness, individual.fidelity, individual.depth), reverse=True)
        fitness_values = [ind.fitness for ind in individuals]
        fidelity_values = [ind.fidelity for ind in individuals]
        depth_values = [ind.depth for ind in individuals]
        diversity = population.calculate_structural_diversity()

        self._data_to_save["fitness_per_generation"].append(fitness_values)
        self._data_to_save["fidelity_per_generation"].append(fidelity_values)
        self._data_to_save["depth_per_generation"].append(depth_values)
        self._data_to_save["structural_diversity_per_generation"].append(diversity)

        # Log no console para feedback imediato
        if generation % 25 == 0:
            avg_fitness = np.mean(fitness_values)
            print(
                f"Generation {generation} | Best Circuit: (Fitness: {fitness_values[0]*100:.10f} ; Fidelity: {fidelity_values[0]*100:.10f} ; "
                f"Depth {depth_values[0]}) | Avg Fitness: {avg_fitness:.4f} | Diversity: {diversity:.4f}"
            )

    def save(self):
        """Salva o dicionário de dados no arquivo JSON."""
        print(f"Saving results to {self._filename}...")
        with open(self._filename, 'w') as f:
            json.dump(self._data_to_save, f, indent=4)
        print("Save complete.")
