import json
from .interfaces import IDataLoader
from .data_models import ResultData


class JsonDataLoader(IDataLoader):
    """
    Carrega dados de resultado de um arquivo JSON.
    É inteligente o suficiente para lidar com múltiplos formatos de dados.
    """

    def load(self, filepath: str) -> ResultData:
        print(f"Carregando dados de {filepath}...")
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Formato 1: Arquivo de resultado de fase (contém a chave 'generations')
        if "generations" in data and isinstance(data["generations"], list):
            generations_data = data["generations"]
            
            return ResultData(
                best_fitness_per_generation=[g.get("best_fitness", 0.0) for g in generations_data],
                average_fitness_per_generation=[g.get("avg_fitness", 0.0) for g in generations_data],
                std_dev_fitness_per_generation=[g.get("std_fitness", 0.0) for g in generations_data],
                
                best_fidelity_per_generation=[g.get("best_fidelity", 0.0) for g in generations_data],
                average_fidelity_per_generation=[g.get("avg_fidelity", 0.0) for g in generations_data],
                std_dev_fidelity_per_generation=[g.get("std_fidelity", 0.0) for g in generations_data],
                
                best_depth_per_generation=[g.get("best_depth", 0) for g in generations_data],
                average_depth_per_generation=[g.get("avg_depth", 0) for g in generations_data],
                std_dev_depth_per_generation=[g.get("std_depth", 0.0) for g in generations_data],
                
                structural_diversity_per_generation=[g.get("structural_diversity", 0.0) for g in generations_data]
            )
        
        # Formato 2: Arquivo já concatenado (chaves correspondem diretamente ao ResultData)
        elif "best_fitness_per_generation" in data:
            # O operador ** desempacota o dicionário nos argumentos do construtor
            return ResultData(**data)
            
        else:
            raise KeyError(f"O arquivo {filepath} não contém um formato de dados reconhecível.")