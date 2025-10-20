import json
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from qiskit.quantum_info import Statevector

from containers import QuantumCircuitContainer
from experiment.config import ExperimentConfig, PhaseConfig
from evolutionary_algorithm.selection import SelectionType
from quantum_circuit.circuit import Circuit
from quantum_circuit.interfaces import IQuantumCircuitAdapter
from shared.value_objects import CrossoverType

PROJECT_PATH = Path(__file__).resolve().parents[3]
TARGET_DIR = PROJECT_PATH / "results" / "target_circuits"


class TestConfigLoader:
    """
    Loads experiment configurations defined in JSON files inside `tests/`.
    Converts them into `ExperimentConfig` and `PhaseConfig` objects.
    """

    def __init__(self, tests_dir: Path):
        self.tests_dir = tests_dir
        if not self.tests_dir.exists():
          raise FileNotFoundError(f"Tests directory not found: {self.tests_dir}")

    @staticmethod
    def _save_circuit_details(circuit: Circuit, adapter: IQuantumCircuitAdapter, filepath_base: str):
        """Saves both JSON representation and ASCII diagram of a circuit."""
        Path(filepath_base).parent.mkdir(parents=True, exist_ok=True)
        with open(f"{filepath_base}.json", "w", encoding="utf-8") as f:
            json.dump(circuit.to_dict(), f, indent=4)
        qiskit_circuit = adapter.from_domain(circuit)
        with open(f"{filepath_base}.txt", "w", encoding="utf-8") as f:
            f.write(str(qiskit_circuit.draw("text")))

    def _load_or_create_target(
            self, num_qubits: int, depth: int, seed_target: int, allowed_gates: Optional[List[str]]
    ) -> Tuple[Statevector, str]:
        """
        Loads an existing target circuit if available, or generates and saves a new one.
        Returns (statevector, filepath_base).
        """
        filepath_base = TARGET_DIR / f"target_seed_{seed_target}"
        filepath_json = Path(f"{filepath_base}.json")

        if filepath_json.exists():
            print(f"📂 Loading existing target circuit from {filepath_json.name}")
            with open(filepath_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            container = QuantumCircuitContainer()
            adapter = container.qiskit_adapter()
            factory = container.circuit_factory()
            circuit = factory.create_from_dict(data)
            qiskit_circuit = adapter.from_domain(circuit)
            target_sv = Statevector.from_instruction(qiskit_circuit)
            return target_sv, str(filepath_base)

        # Otherwise: generate deterministically
        print(f"⚙️ Generating new target circuit (seed={seed_target})...")
        random.seed(seed_target)
        np.random.seed(seed_target)

        container = QuantumCircuitContainer()
        container.config.from_dict({"quantum": {"allowed_gates": allowed_gates}})

        factory = container.circuit_factory()
        adapter = container.qiskit_adapter()

        domain_circuit = factory.create_random_circuit(
            num_qubits=num_qubits,
            max_depth=depth,
            min_depth=depth,
            use_evolutionary_strategy=False
        )

        # Save it
        self._save_circuit_details(domain_circuit, adapter, str(filepath_base))

        qiskit_circuit = adapter.from_domain(domain_circuit)
        target_sv = Statevector.from_instruction(qiskit_circuit)
        return target_sv, str(filepath_base)

    @staticmethod
    def _load_json(file_path: Path) -> dict:
        """Safely loads a JSON configuration file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _build_phase(phase_dict: dict) -> PhaseConfig:
        """Constructs a PhaseConfig from its dict representation."""
        return PhaseConfig(
            use_stepsize=phase_dict["use_stepsize"],
            use_weighted_fitness=phase_dict["use_weighted_fitness"],
            use_adaptive_rates=phase_dict["use_adaptive_rates"],
            use_bandit_mutation=phase_dict["use_bandit_mutation"],
            parent_selection=SelectionType[phase_dict["parent_selection"].upper()],
            survivor_selection=SelectionType[phase_dict["survivor_selection"].upper()],
            use_fitness_sharing=phase_dict["use_fitness_sharing"],
            crossover_strategy=phase_dict["crossover_strategy"].lower(),
            generations=int(phase_dict["generations"]),
            fidelity_threshold_stop=phase_dict.get("fidelity_threshold_stop"),
        )

    def _build_experiment(self, cfg: dict) -> ExperimentConfig:
        """Constructs an ExperimentConfig, filling only required fields."""
        phases = [self._build_phase(p) for p in cfg["phases"]]

        target_sv, target_filepath = self._load_or_create_target(
            num_qubits=cfg.get("num_qubits", ExperimentConfig.num_qubits),
            depth=cfg["target_depth"],
            seed_target=cfg["seed_target"],
            allowed_gates=cfg.get("allowed_gates"),
        )

        # --- Campos obrigatórios ---
        required_kwargs = dict(
            seed=cfg["seed"],
            max_depth=cfg["max_depth"],
            min_depth=cfg["min_depth"],
            target_depth=cfg["target_depth"],
            target_statevector_data=target_sv,
            filename_target_circuit=target_filepath,
            phases=phases,
            resume_from_checkpoint=cfg["resume_from_checkpoint"],
        )

        # --- Campos opcionais (se existirem, sobrescrevem os defaults do dataclass) ---
        optional_keys = [
            "allowed_gates", "num_qubits", "elitism_size", "population_size",
            "tournament_size", "crossover_rate", "mutation_rate",
            "min_mutation_rate", "max_mutation_rate",
            "min_crossover_rate", "max_crossover_rate",
            "diversity_threshold", "injection_rate",
            "sharing_radius", "alpha", "c_factor"
        ]
        for key in optional_keys:
            if key in cfg:
                required_kwargs[key] = cfg[key]

        return ExperimentConfig(**required_kwargs)

    def load_all(self) -> Tuple[List[ExperimentConfig], List[str]]:
        """Loads all JSON configs from the tests directory."""
        configs = []
        filenames = []
        for file_path in sorted(self.tests_dir.glob("*.json")):
            try:
                cfg = self._load_json(file_path)
                configs.append(self._build_experiment(cfg))
                filenames.append(file_path.name)
                print(f"✅ Loaded test config: {file_path.name}")
            except Exception as e:
                print(f"⚠️ Failed to load {file_path.name}: {e}")
        return configs, filenames
