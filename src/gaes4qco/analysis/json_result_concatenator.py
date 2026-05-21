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

    def _concat_result_data(self, result_data_list: List[ResultData]) -> ResultData:
        """
        Concatena dados de várias phases em um único ResultData contínuo.
        """
        fitness_all = []
        diversity_all = []
        fidelity_all = []
        depth_all = []
        for rd in result_data_list:
            fitness_all.extend(rd.fitness_per_generation)
            diversity_all.extend(rd.structural_diversity_per_generation)
            fidelity_all.extend(rd.fidelity_per_generation)
            depth_all.extend(rd.depth_per_generation)

        return ResultData(
            fitness_per_generation=fitness_all,
            structural_diversity_per_generation=diversity_all,
            fidelity_per_generation=fidelity_all,
            depth_per_generation=depth_all
        )

    def process_single_test(self, config: ExperimentConfig, test_filename: str) -> Path:
        """
        Gera um único arquivo concatenado para um test.json específico.
        """
        result_files = []
        for p in config.phases:
            if p.result_filepath:
                abs_path = PROJECT_PATH / p.result_filepath
                if abs_path.exists():
                    result_files.append(abs_path)
                else:
                    print(f"⚠️ Aviso: Arquivo de resultado não encontrado: {abs_path}")
        
        if not result_files:
            print(f"⚠️ Nenhum resultado válido encontrado para {test_filename}")
            return None

        print(f"🔗 Concatenando {len(result_files)} fases para {test_filename}...")

        # Carrega todos os dados
        result_data_list = [self.data_loader.load(str(f)) for f in result_files]

        # Concatena
        concatenated = self._concat_result_data(result_data_list)

        # Salva em um novo arquivo
        output_path = self.output_dir / f"{test_filename.replace('.json', '')}_concatenated_result.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "fitness_per_generation": concatenated.fitness_per_generation,
                "structural_diversity_per_generation": concatenated.structural_diversity_per_generation,
                "fidelity_per_generation": concatenated.fidelity_per_generation,
                "depth_per_generation": concatenated.depth_per_generation
            }, f, indent=4)

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
