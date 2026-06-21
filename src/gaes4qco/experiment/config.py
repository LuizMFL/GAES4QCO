import enum
from dataclasses import dataclass, asdict, is_dataclass
from enum import Enum
from typing import List, Any, Optional, Generator
from pathlib import Path
import json
import hashlib

from evolutionary_algorithm.selection import SelectionType
from shared.value_objects import CrossoverType

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class FitnessEvaluatorType(Enum):
    DEFAULT = "default"
    WEIGHTED = "weighted"
    CNOT_PENALTY = "cnot_penalty"


@dataclass
class PhaseConfig:
    """Configuração para uma única fase da otimização."""
    use_stepsize: bool
    fitness_evaluator: FitnessEvaluatorType
    use_adaptive_rates: bool
    use_bandit_mutation: bool
    parent_selection: SelectionType
    survivor_selection: SelectionType
    use_fitness_sharing: bool
    crossover_strategy: CrossoverType
    generations: int
    fidelity_threshold_stop: Optional[float]

    crossover_rate: Optional[float] = None
    mutation_rate: Optional[float] = None
    min_mutation_rate: Optional[float] = None
    max_mutation_rate: Optional[float] = None
    min_crossover_rate: Optional[float] = None
    max_crossover_rate: Optional[float] = None

    result_filepath: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Encapsula todos os parâmetros para uma única execução do GA."""
    seed: int
    target_statevector_data: List[Any]
    filename_target_circuit: str
    phases: List[PhaseConfig]
    resume_from_checkpoint: bool

    max_depth: int = 40
    min_depth: int = 4
    allowed_gates: Optional[List[str]] = None
    target_depth: int = 20
    num_qubits: int = 4
    elitism_size: int = 10
    population_size: int = 200
    tournament_size: Optional[int] = None
    sharing_radius: Optional[float] = None
    alpha: Optional[float] = None
    c_factor: Optional[float] = None

    def __post_init__(self):
        uses_tournament = any(
            p.parent_selection == SelectionType.TOURNAMENT or
            p.survivor_selection == SelectionType.TOURNAMENT
            for p in self.phases
        )
        uses_fitness_sharing = any(p.use_fitness_sharing for p in self.phases)
        uses_stepsize = any(p.use_stepsize for p in self.phases)

        for i, p in enumerate(self.phases):
            if not p.use_adaptive_rates and (p.crossover_rate is None or p.mutation_rate is None):
                raise ValueError(f"crossover_rate e mutation_rate são exigidos na fase {i} (taxas fixas).")
            if p.use_adaptive_rates and None in (p.min_mutation_rate, p.max_mutation_rate, p.min_crossover_rate,
                                                 p.max_crossover_rate):
                raise ValueError(f"min/max rates são exigidos na fase {i} (taxas adaptativas).")

        if uses_tournament and self.tournament_size is None:
            raise ValueError("tournament_size é exigido para seleção de torneio.")
        if uses_fitness_sharing and (self.sharing_radius is None or self.alpha is None):
            raise ValueError("sharing_radius e alpha são exigidos para fitness sharing.")
        if uses_stepsize and self.c_factor is None:
            raise ValueError("c_factor é exigido para stepsize.")

    def get_config_foldername(self) -> Generator[str, Any, None]:
        for i, phase in enumerate(self.phases):
            if phase.fitness_evaluator == FitnessEvaluatorType.WEIGHTED:
                fit_flag = "WG"
            elif phase.fitness_evaluator == FitnessEvaluatorType.CNOT_PENALTY:
                fit_flag = "CP"
            else:
                fit_flag = "FD"  # Default (Fidelity)
            rate_flag = "AD" if phase.use_adaptive_rates else "FX"
            mut_flag = "BD" if phase.use_bandit_mutation else "RD"
            step_flag = "ST" if phase.use_stepsize else "NR"
            select_parent_flag = phase.parent_selection.value[:2]
            select_survivor_flag = phase.survivor_selection.value[0:2]
            fit_shaper_flag = "FT" if phase.use_fitness_sharing else "NL"
            crossover_flag = phase.crossover_strategy[0:2]
            yield f"pha={i}_{fit_flag}_{crossover_flag}_{select_parent_flag}_{select_survivor_flag}_{rate_flag}_{mut_flag}_{step_flag}_{fit_shaper_flag}"

    def get_config_hash(self) -> Generator[str, Any, None]:
        data = asdict(self)
        data.pop("target_statevector_data", None)
        data.pop("resume_from_checkpoint", None)
        if "phases" in data:
            for phase in data["phases"]:
                phase.pop("result_filepath", None)

        def custom_serializer(o):
            if is_dataclass(o): return asdict(o)
            if isinstance(o, enum.Enum): return o.value
            raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

        for i in range(1, len(self.phases) + 1):
            data_final = data.copy()
            data_final["phases"] = data_final["phases"][:i]
            canonical_string = json.dumps(data_final, sort_keys=True, separators=(",", ":"), default=custom_serializer)
            hasher = hashlib.sha256(canonical_string.encode("utf-8"))
            yield hasher.hexdigest()[:8]

    @property
    def config_file_path(self) -> Generator[str, Any, None]:
        folder_path = PROJECT_ROOT / "results"
        for i, (config_foldername, config_hash) in enumerate(zip(self.get_config_foldername(), self.get_config_hash())):
            folder_path = folder_path / config_foldername
            yield str(folder_path / f"{config_hash}_config.json")

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("target_statevector_data", None)
        data.pop("resume_from_checkpoint", None)
        data.pop("phases", None)
        data.pop("config_file_path", None)
        
        keys_to_remove = [k for k, v in data.items() if v is None]
        for k in keys_to_remove:
            data.pop(k, None)
            
        return data
