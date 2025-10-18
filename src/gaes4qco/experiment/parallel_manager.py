import time
from dataclasses import asdict
from multiprocessing import Pool, cpu_count
from typing import List

from containers import ExperimentContainer
from .config import ExperimentConfig
from .runner import ExperimentRunner


class ParallelExperimentManager:
    """Gerencia a execução de múltiplos experimentos em paralelo."""

    def __init__(self, configs: List[ExperimentConfig], filenames: List[str], max_processes: int):
        self.configs = configs
        self.filenames = filenames
        self.max_processes = min(max_processes, cpu_count())
        self.experiment_container = ExperimentContainer()

    def run_all(self) -> List[dict]:
        """Executa todos os experimentos configurados usando um pool de processos."""
        print(f"Iniciando {len(self.configs)} experimentos em {self.max_processes} processos paralelos...")
        start_time = time.time()
        experiments = []
        for i, cfg in enumerate(self.configs):
            config_dict = asdict(cfg)
            self.experiment_container.config.from_dict(config_dict)
            runner = self.experiment_container.runner()
            experiments.append((runner, self.filenames[i]))

        with Pool(self.max_processes) as pool:
            # Usa starmap para passar cada objeto de configuração para a função de execução
            results = pool.starmap(run_experiment, experiments)

        total_duration = time.time() - start_time
        print(f"--- Fim de todos os experimentos | Duração Total: {total_duration:.2f}s ---")

        return results


def run_experiment(runner: ExperimentRunner, filename: str) -> dict:
    """Executa um experimento e anexa o nome do arquivo de origem."""
    result = runner.run()
    result["filename"] = filename
    return result
