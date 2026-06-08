import json
import logging
import numpy as np

from quantum_circuit.circuit import Circuit
from .interfaces import IProgressObserver
from evolutionary_algorithm.population import Population


class JsonProgressObserver(IProgressObserver):
    """
    Implementa um observador que coleta estatísticas e salva em um arquivo JSON.
    """

    def __init__(self, filename: str, test_filename: str = ""):
        self._filename = filename
        self._test_filename = test_filename
        self._generations_data = []
        self._summary = {}

    def update(self, generation: int, population: Population, mutation_rate: float, crossover_rate: float):
        """Coleta os dados consolidados da população atual."""
        evaluated_individuals = [ind for ind in population.get_individuals() if ind.fitness is not None]
        if not evaluated_individuals:
            return

        best_individual = max(evaluated_individuals, key=lambda ind: ind.base_fitness)
        
        fitness_values = [ind.fitness for ind in evaluated_individuals]
        fidelity_values = [ind.fidelity for ind in evaluated_individuals if ind.fidelity is not None]
        depth_values = [ind.depth for ind in evaluated_individuals]
        
        diversity = population.calculate_structural_diversity()

        gen_record = {
            "generation": generation,
            "best_fitness": best_individual.fitness,
            "best_fidelity": best_individual.fidelity,
            "best_depth": best_individual.depth,
            "avg_fitness": float(np.mean(fitness_values)) if fitness_values else 0.0,
            "std_fitness": float(np.std(fitness_values)) if fitness_values else 0.0,
            "avg_fidelity": float(np.mean(fidelity_values)) if fidelity_values else 0.0,
            "std_fidelity": float(np.std(fidelity_values)) if fidelity_values else 0.0,
            "avg_depth": float(np.mean(depth_values)) if depth_values else 0.0,
            "std_depth": float(np.std(depth_values)) if depth_values else 0.0,
            "structural_diversity": diversity,
            "mutation_rate": mutation_rate,
            "crossover_rate": crossover_rate
        }
        self._generations_data.append(gen_record)

        if generation == 0 or generation % 25 == 0:
            log_prefix = f"[{self._test_filename}] " if self._test_filename else ""
            logging.info(
                f"{log_prefix}Gen {generation:04d} | Best [Fit: {gen_record['best_fitness']:.4f} | Fid: {gen_record['best_fidelity']:.4f} | Dep: {gen_record['best_depth']:03d}] "
                f"| Pop Avg Fit: {gen_record['avg_fitness']:.4f} | Div: {diversity:.4f}"
            )

    def set_summary(self, duration_seconds: float, final_generation: int, total_evaluations: int, stopping_reason: str,
                    best_circuit: Circuit):
        self._summary = {
            "phase_duration_seconds": duration_seconds,
            "total_generations_executed": final_generation,
            "total_fitness_evaluations": total_evaluations,
            "stopping_reason": stopping_reason,
            "best_circuit_filename": f"rank_000_fit_{best_circuit.fitness:.4f}_fid_{best_circuit.fidelity:.4f}_depth_{best_circuit.depth}.json"
        }

    def save(self):
        if not self._generations_data:
            logging.warning(f"⚠️ Nenhum dado de geração para salvar em {self._filename}.")
            return

        data_to_save = {
            "generations": self._generations_data,
            "summary": self._summary
        }

        logging.info(f"Saving results to {self._filename}...")
        with open(self._filename, 'w') as f:
            json.dump(data_to_save, f, indent=4)
        logging.info("Save complete.")