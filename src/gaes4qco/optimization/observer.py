# optimization/observer.py

import json
import numpy as np

from quantum_circuit.circuit import Circuit
from .interfaces import IProgressObserver
from evolutionary_algorithm.population import Population


class JsonProgressObserver(IProgressObserver):
    """
    ## Implementa um observador que coleta estatísticas e salva em um arquivo JSON.
    """

    def __init__(self, filename: str):
        self._filename = filename
        self._data_to_save = {
            "summary": {
                "phase_duration_seconds": 0.0,
                "total_generations_executed": 0,
                "total_fitness_evaluations": 0,
                "stopping_reason": "",
                "best_circuit_filename": "",
            },
            "generations": []
        }

    def update(self, generation: int, population: Population, mutation_rate: float = 0.0, crossover_rate: float = 0.0):
        """Coleta os dados consolidados da população atual."""
        individuals = population.get_individuals()
        individuals.sort(key=lambda individual: (individual.fitness, individual.fidelity, individual.depth), reverse=True)

        best_individual = individuals[0]

        fitness_values = [ind.fitness for ind in individuals]
        fidelity_values = [ind.fidelity for ind in individuals]
        diversity = population.calculate_structural_diversity()

        gen_record = {
            "generation": generation,

            # 1. Métricas exatas do MELHOR indivíduo
            "best_fitness": best_individual.fitness,
            "best_fidelity": best_individual.fidelity,
            "best_depth": best_individual.depth,
            "best_cx_count": best_individual.get_cx_count(), # NOVO CAMPO ATIVO

            # 2. Estatísticas da População
            "avg_fitness": float(np.mean(fitness_values)),
            "std_fitness": float(np.std(fitness_values)),
            "avg_fidelity": float(np.mean(fidelity_values)),

            # 3. Comportamento do Algoritmo
            "structural_diversity": diversity,
            "mutation_rate": mutation_rate,
            "crossover_rate": crossover_rate
        }

        self._data_to_save["generations"].append(gen_record)

        if generation % 25 == 0:
            print(
                f"Gen {generation:04d} | Best [Fit: {gen_record['best_fitness']:.4f} | Fid: {gen_record['best_fidelity']:.4f} | Dep: {gen_record['best_depth']:03d} | CX: {gen_record['best_cx_count']:02d}] "
                f"| Pop Avg Fit: {gen_record['avg_fitness']:.4f} | Div: {diversity:.4f}"
            )

    def set_summary(self, duration_seconds: float, final_generation: int, total_evaluations: int, stopping_reason: str,
                    best_circuit: Circuit):
        """Registra o resumo final e vital da fase."""
        self._data_to_save["summary"]["phase_duration_seconds"] = duration_seconds
        self._data_to_save["summary"]["total_generations_executed"] = final_generation
        self._data_to_save["summary"]["total_fitness_evaluations"] = total_evaluations
        self._data_to_save["summary"]["stopping_reason"] = stopping_reason

        # Reconstrói o padrão de nome de arquivo que o Runner usa para salvar os circuitos
        best_filename = f"rank_000_fit_{best_circuit.fitness:.4f}_fid_{best_circuit.fidelity:.4f}_depth_{best_circuit.depth}.json"
        self._data_to_save["summary"]["best_circuit_filename"] = best_filename

    def set_duration(self, duration_seconds: float, final_generation: int):
        """Registra o resumo final da fase."""
        self._data_to_save["summary"]["phase_duration_seconds"] = duration_seconds
        self._data_to_save["summary"]["total_generations_executed"] = final_generation

    def save(self):
        """Salva o dicionário de dados no arquivo JSON."""
        print(f"Saving results to {self._filename}...")
        with open(self._filename, 'w') as f:
            json.dump(self._data_to_save, f, indent=4)
        print("Save complete.")
