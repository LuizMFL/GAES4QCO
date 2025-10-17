import json
from .interfaces import IDataLoader
from .data_models import ResultData


class JsonDataLoader(IDataLoader):
    """Carrega dados de resultado de um arquivo JSON."""

    def load(self, filepath: str) -> ResultData:
        print(f"Carregando dados de {filepath}...")
        with open(filepath, 'r') as f:
            data = json.load(f)

        return ResultData(
            fitness_per_generation=data["fitness_per_generation"],
            structural_diversity_per_generation=data["structural_diversity_per_generation"],
            fidelity_per_generation=data["fidelity_per_generation"],
            depth_per_generation=data.get("depth_per_generation", [])
        )
