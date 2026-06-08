import json
from pathlib import Path
from typing import Optional

from evolutionary_algorithm.population_factory import PopulationFactory
from experiment.config import ExperimentConfig
from evolutionary_algorithm.population import Population


class CheckpointManager:
    """Gerencia o salvamento e carregamento de checkpoints da população."""

    def __init__(self, config: ExperimentConfig, population_factory: PopulationFactory):
        self.config = config
        self._phase_checkpoint_path: Optional[Path] = None
        self._population_factory = population_factory

    def phase_checkpoint_exists(self) -> bool:
        """Verifica se o checkpoint da Fase 1 já existe."""
        if self._phase_checkpoint_path is None:
            return False
        return self._phase_checkpoint_path.is_dir()

    def save_phase_checkpoint(self, population: Population):
        """Salva a população final da Fase 1 como um checkpoint JSON."""
        print(f"Salvando checkpoint da Fase 1 em: {self._phase_checkpoint_path}")
        population_dict = {
            "individuals": [ind.to_dict() for ind in population.get_individuals()]
        }
        with open(self._phase_checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(population_dict, f, indent=4)

    def load_phase_checkpoint(self, path: Path) -> Optional[Population]:
        """Carrega e reconstrói a população a partir de um checkpoint JSON."""
        self._phase_checkpoint_path = path

        if not self.phase_checkpoint_exists():
            return Population()

        circuits_dicts = []
        print(f"Carregando checkpoint da Fase 1 de: {self._phase_checkpoint_path}")
        for file in path.iterdir():
            if not file.suffix == '.json':
                continue
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                circuits_dicts.append(data)
        print(f"Carregado com sucesso!")
        return self._population_factory.create_from_list_dict(circuits_dicts)
