import random
import time
import json
import logging
from pathlib import Path
from typing import List

import numpy as np

from evolutionary_algorithm.population import Population
from quantum_circuit.circuit import Circuit
from quantum_circuit.interfaces import IQuantumCircuitAdapter
from .config import ExperimentConfig, PhaseConfig


def save_circuit_details(circuit: Circuit, adapter: IQuantumCircuitAdapter, filepath_base: str):
    """Salva a estrutura de um circuito em .json e sua representação em .txt."""
    with open(f"{filepath_base}.json", 'w', encoding='utf-8') as f:
        json.dump(circuit.to_dict(), f, indent=4)
    qiskit_circuit = adapter.from_domain(circuit)
    with open(f"{filepath_base}.txt", 'w', encoding='utf-8') as f:
        f.write(str(qiskit_circuit.draw('text')))


def circuits_folder_path(config_file_path: Path) -> Path:
    return Path(str(config_file_path).replace("_config.json", "_circuits"))


def save_final_population(circuits: List[Circuit], adapter: IQuantumCircuitAdapter, config_file_path: Path):
    """Salva uma lista de circuitos em uma subpasta dedicada."""
    folder_path = circuits_folder_path(config_file_path)
    folder_path.mkdir(parents=True, exist_ok=True)

    logging.info(f"Salvando {len(circuits)} circuitos finais em '{folder_path}'...")

    circuits.sort(key=lambda c: c.fitness, reverse=True)
    for i, circuit in enumerate(circuits):
        basename = f"rank_{i:03d}_fit_{circuit.fitness:.4f}_fid_{circuit.fidelity:.4f}_depth_{circuit.depth}"
        filepath_base = str(folder_path / basename)
        save_circuit_details(circuit, adapter, filepath_base)


class ExperimentRunner:
    """Executa uma única instância completa de um experimento do GA."""

    def __init__(self, config: dict, test_filename: str, container):
        if "phases" in config and isinstance(config["phases"], list):
            for i, phase in enumerate(config["phases"]):
                if isinstance(phase, dict):
                    config["phases"][i] = PhaseConfig(**phase)
        self.config = ExperimentConfig(**config)
        self.test_filename = test_filename
        self.container = container()

    def _configure_container_for_phase(self, phase_config: PhaseConfig, observer_filename: str):
        self.container.config.from_dict({
            "quantum": {
                "target_statevector_data": self.config.target_statevector_data,
                "num_qubits": self.config.num_qubits,
                "target_depth": self.config.target_depth,
                "allowed_gates": self.config.allowed_gates
            },
            "selection_strategy": {
                "fitness": "weighted" if phase_config.use_weighted_fitness else "default",
                "fitness_shaper": "sharing" if phase_config.use_fitness_sharing else "default",
                "rate_adapter": "adaptive" if phase_config.use_adaptive_rates else "default",
                "mutation": "bandit" if phase_config.use_bandit_mutation else "default",
                "parent_selection": phase_config.parent_selection.value if hasattr(phase_config.parent_selection, 'value') else phase_config.parent_selection,
                "survivor_selection": phase_config.survivor_selection.value if hasattr(phase_config.survivor_selection, 'value') else phase_config.survivor_selection,
                "crossover": phase_config.crossover_strategy
            },
            "evolution": {
                "population_size": self.config.population_size,
                "elitism_size": self.config.elitism_size,
                "tournament_size": self.config.tournament_size,
                "crossover_rate": self.config.crossover_rate,
                "mutation_rate": self.config.mutation_rate,
                "max_depth": self.config.max_depth,
                "diversity_threshold": self.config.diversity_threshold,
                "injection_rate": self.config.injection_rate,
                "stepsize": phase_config.use_stepsize,
                "c_factor": self.config.c_factor
            },
            "adaptive_rates": {
                "min_mutation_rate": self.config.min_mutation_rate,
                "max_mutation_rate": self.config.max_mutation_rate,
                "min_crossover_rate": self.config.min_crossover_rate,
                "max_crossover_rate": self.config.max_crossover_rate,
            },
            "niching": {
                "sharing_radius": self.config.sharing_radius,
                "alpha": self.config.alpha
            },
            "observer": {
                "filename": observer_filename,
                "test_filename": self.test_filename # Passa o nome do arquivo para o container
            }
        })

    def run(self) -> dict:
        logging.info(f"--- Iniciando Experimento com Seed {self.config.seed} ---")
        start_time = time.time()

        random.seed(self.config.seed)
        np.random.seed(self.config.seed)

        population: Population = Population()
        result_files = []
        for i, (phase, config_file_path_str) in enumerate(zip(self.config.phases, self.config.config_file_path)):
            logging.info(f"--- FASE {i} ---")
            config_file_path = Path(config_file_path_str)
            logging.info(f"Salvando configuração do experimento em: {config_file_path}")
            Path(config_file_path).parent.mkdir(parents=True, exist_ok=True)
            with open(config_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.config.to_dict(), f, indent=4)
            results_file_path = str(config_file_path).replace("_config.json", "_results.json")
            result_files.append(results_file_path)
            self._configure_container_for_phase(phase, results_file_path)

            if self.config.resume_from_checkpoint:
                checkpoint_manager = self.container.checkpoint_manager()
                population_temp = checkpoint_manager.load_phase_checkpoint(Path(circuits_folder_path(config_file_path)))
                if population_temp.get_individuals():
                    population = population_temp
                    continue
            if not population.get_individuals():
                pop_factory = self.container.population_fac()
                population = pop_factory.create(
                    self.config.population_size, self.config.num_qubits,
                    self.config.max_depth, self.config.min_depth, phase.use_stepsize
                )

            optimizer = self.container.optimizer()
            population = optimizer.run(population, phase.generations, phase.fidelity_threshold_stop)

            logging.info("Optimization finished.")
            final_circuits = population.get_individuals()

            adapter = self.container.circuit.qiskit_adapter()
            save_final_population(final_circuits, adapter, config_file_path)

        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"--- Fim Experimento Seed {self.config.seed} | Duração: {duration:.2f}s ---")

        best_circuit = population.get_fittest()
        return {
            "seed": self.config.seed,
            "best_fitness": best_circuit.fitness if best_circuit else 0.0,
            "duration_seconds": duration,
            "result_files": result_files
        }