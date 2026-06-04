# src/analysis/result_concatenator.py

import json
from pathlib import Path
from typing import List

from analysis.interfaces import IJsonResultConcatenator
from experiment.test_loader import TestConfigLoader
from experiment.config import ExperimentConfig
from analysis.loader import JsonDataLoader
from analysis.data_models import ResultData

PROJECT_PATH = Path(__file__).resolve().parents[3]


class JsonResultConcatenator(IJsonResultConcatenator):
    """
    Combina os resultados de múltiplas phases de um mesmo experimento em um único JSON.
    """

    def __init__(self, tests_dir: Path, results_dir: Path):
        self.tests_dir = tests_dir
        self.results_dir = results_dir

        self.loader = TestConfigLoader(self.tests_dir)
        self.data_loader = JsonDataLoader()

        self.output_dir = self.results_dir / "concatenated"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _reconstruct_result_paths(self, config: ExperimentConfig) -> List[Path]:
        """
        Usa a mesma lógica do ExperimentConfig para reconstruir os caminhos dos arquivos de resultado.
        """
        paths = []
        folder_names = list(config.get_config_foldername())
        hashes = list(config.get_config_hash())

        current_path = self.results_dir
        for i in range(len(config.phases)):
            folder_name = folder_names[i]
            config_hash = hashes[i]
            
            phase_path = current_path / folder_name
            result_file = phase_path / f"{config_hash}_results.json"
            
            if result_file.exists():
                paths.append(result_file)
            else:
                print(f"⚠️ Aviso: Arquivo de resultado não encontrado: {result_file}")
            
            current_path = phase_path
            
        return paths

    def _concat_result_data(self, result_data_list: List[ResultData]) -> ResultData:
        """
        Concatena os dados de séries temporais de várias fases em um único ResultData.
        """
        concatenated_data = ResultData()
        for rd in result_data_list:
            concatenated_data.best_fitness_per_generation.extend(rd.best_fitness_per_generation)
            concatenated_data.average_fitness_per_generation.extend(rd.average_fitness_per_generation)
            concatenated_data.std_dev_fitness_per_generation.extend(rd.std_dev_fitness_per_generation)
            
            concatenated_data.best_fidelity_per_generation.extend(rd.best_fidelity_per_generation)
            concatenated_data.average_fidelity_per_generation.extend(rd.average_fidelity_per_generation)
            concatenated_data.std_dev_fidelity_per_generation.extend(rd.std_dev_fidelity_per_generation)

            concatenated_data.best_depth_per_generation.extend(rd.best_depth_per_generation)
            concatenated_data.average_depth_per_generation.extend(rd.average_depth_per_generation)
            concatenated_data.std_dev_depth_per_generation.extend(rd.std_dev_depth_per_generation)
            
            concatenated_data.structural_diversity_per_generation.extend(rd.structural_diversity_per_generation)
        return concatenated_data

    def process_single_test(self, config: ExperimentConfig, test_filename: str) -> Path:
        """
        Gera um único arquivo concatenado para um test.json específico.
        """
        result_files = self._reconstruct_result_paths(config)
        
        if not result_files:
            print(f"⚠️ Nenhum resultado válido encontrado para {test_filename}")
            return None

        print(f"🔗 Concatenando {len(result_files)} fases para {test_filename}...")

        result_data_list = [self.data_loader.load(str(f)) for f in result_files]
        concatenated = self._concat_result_data(result_data_list)

        output_path = self.output_dir / f"{Path(test_filename).stem}_concatenated_result.json"
        with open(output_path, "w", encoding="utf-8") as f:
            from dataclasses import asdict
            json.dump(asdict(concatenated), f, indent=4)

        print(f"✅ Resultado concatenado salvo em: {output_path.name}")
        return output_path

    def process_all_tests(self) -> List[Path]:
        """
        Processa todos os arquivos de teste encontrados no diretório `tests/`
        e retorna a lista de caminhos dos arquivos concatenados gerados.
        """
        experiment_configs, filenames = self.loader.load_all(update_json=False)
        concatenated_paths = []

        for cfg, fname in zip(experiment_configs, filenames):
            path = self.process_single_test(cfg, fname)
            if path:
                concatenated_paths.append(path)

        print(f"\n📦 {len(concatenated_paths)} arquivos concatenados gerados com sucesso.")
        return concatenated_paths
