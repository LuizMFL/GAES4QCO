import enum
from dataclasses import dataclass, field, asdict, is_dataclass
from typing import List, Any, Optional, Generator
from pathlib import Path
import json
import hashlib

from evolutionary_algorithm.selection import SelectionType
from shared.value_objects import CrossoverType

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class PhaseConfig:
    """Configuração para uma única fase da otimização."""
    use_stepsize: bool
    use_weighted_fitness: bool
    use_adaptive_rates: bool
    use_bandit_mutation: bool
    parent_selection: SelectionType
    survivor_selection: SelectionType
    use_fitness_sharing: bool
    crossover_strategy: CrossoverType
    generations: int
    fidelity_threshold_stop: Optional[float]
    result_filepath: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Encapsula todos os parâmetros para uma única execução do GA."""
    seed: int
    max_depth: int
    min_depth: int
    target_statevector_data: List[Any]
    filename_target_circuit: str
    phases: List[PhaseConfig]
    resume_from_checkpoint: bool
    allowed_gates: Optional[List[str]] = None
    target_depth: int = 20
    num_qubits: int = 4
    elitism_size: int = 10
    population_size: int = 200
    diversity_threshold: float = 0.1  # Limiar de 10%
    injection_rate: float = 0.15  # Injeta 15% quando ativado
    
    # Hiperparâmetros Condicionais
    tournament_size: Optional[int] = None
    crossover_rate: Optional[float] = None
    mutation_rate: Optional[float] = None
    min_mutation_rate: Optional[float] = None
    max_mutation_rate: Optional[float] = None
    min_crossover_rate: Optional[float] = None
    max_crossover_rate: Optional[float] = None
    sharing_radius: Optional[float] = None
    alpha: Optional[float] = None
    c_factor: Optional[float] = None

    def __post_init__(self):
        # --- Validação e Limpeza de Hiperparâmetros Condicionais ---
        uses_fixed_rates = any(not p.use_adaptive_rates for p in self.phases)
        uses_adaptive_rates = any(p.use_adaptive_rates for p in self.phases)
        uses_tournament = any(
            p.parent_selection == SelectionType.TOURNAMENT or 
            p.survivor_selection == SelectionType.TOURNAMENT 
            for p in self.phases
        )
        uses_fitness_sharing = any(p.use_fitness_sharing for p in self.phases)
        uses_stepsize = any(p.use_stepsize for p in self.phases)

        # 1. Fixed Rates
        if uses_fixed_rates:
            if self.crossover_rate is None or self.mutation_rate is None:
                raise ValueError("crossover_rate e mutation_rate são exigidos quando use_adaptive_rates é False em qualquer fase.")
        else:
            self.crossover_rate = None
            self.mutation_rate = None

        # 2. Adaptive Rates
        if uses_adaptive_rates:
            if None in (self.min_mutation_rate, self.max_mutation_rate, self.min_crossover_rate, self.max_crossover_rate):
                raise ValueError("min_mutation_rate, max_mutation_rate, min_crossover_rate e max_crossover_rate são exigidos quando use_adaptive_rates é True em qualquer fase.")
        else:
            self.min_mutation_rate = None
            self.max_mutation_rate = None
            self.min_crossover_rate = None
            self.max_crossover_rate = None

        # 3. Tournament Selection
        if uses_tournament:
            if self.tournament_size is None:
                raise ValueError("tournament_size é exigido quando parent_selection ou survivor_selection é TOURNAMENT em qualquer fase.")
        else:
            self.tournament_size = None

        # 4. Fitness Sharing
        if uses_fitness_sharing:
            if self.sharing_radius is None or self.alpha is None:
                raise ValueError("sharing_radius e alpha são exigidos quando use_fitness_sharing é True em qualquer fase.")
        else:
            self.sharing_radius = None
            self.alpha = None

        # 5. Step Size
        if uses_stepsize:
            if self.c_factor is None:
                raise ValueError("c_factor é exigido quando use_stepsize é True em qualquer fase.")
        else:
            self.c_factor = None


    def get_config_foldername(self) -> Generator[str, Any, None]:
        """Gera um nome de pasta descritivo a partir das flags de configuração."""
        for i, phase in enumerate(self.phases):
            fit_flag = "WG" if phase.use_weighted_fitness else "FD"  # Weighted vs Fidelity-only
            rate_flag = "AD" if phase.use_adaptive_rates else "FX"  # Adaptive vs Fixed
            mut_flag = "BD" if phase.use_bandit_mutation else "RD"  # Bandit vs Random
            step_flag = "ST" if phase.use_stepsize else "NR"  # Stepsize vs Normal
            select_parent_flag = phase.parent_selection.value[:2]
            select_survivor_flag = phase.survivor_selection.value[0:2]
            fit_shaper_flag = "FT" if phase.use_fitness_sharing else "NL"  # Fitness Sharing Shaper vs Null Fitness Shaper
            crossover_flag = phase.crossover_strategy[0:2]
            yield f"pha={i}_{fit_flag}_{crossover_flag}_{select_parent_flag}_{select_survivor_flag}_{rate_flag}_{mut_flag}_{step_flag}_{fit_shaper_flag}"

    def get_config_hash(self) -> Generator[str, Any, None]:
        """
        Gera um hash SHA256 curto e único para a configuração do experimento.
        """
        data = asdict(self).copy()
        data.pop("target_statevector_data", None)
        data.pop("resume_from_checkpoint", None)

        if "phases" in data:
            for phase in data["phases"]:
                phase.pop("result_filepath", None)

        def custom_serializer(o):
            if is_dataclass(o):
                return asdict(o)  # converte dataclass para dict
            if isinstance(o, enum.Enum):
                return o.value  # ou .value, dependendo do que você quer
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
        """Converte a configuração para um dicionário, excluindo dados grandes."""
        data = asdict(self)
        del data["target_statevector_data"]
        del data["resume_from_checkpoint"]
        del data["phases"]
        data.pop("config_file_path", None)
        
        keys_to_remove = [k for k, v in data.items() if v is None and k in [
            "tournament_size", "crossover_rate", "mutation_rate", 
            "min_mutation_rate", "max_mutation_rate", "min_crossover_rate", 
            "max_crossover_rate", "sharing_radius", "alpha", "c_factor"
        ]]
        for k in keys_to_remove:
            data.pop(k, None)
            
        return data