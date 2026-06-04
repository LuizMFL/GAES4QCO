import time
from dataclasses import asdict
from multiprocessing import Pool, cpu_count
from typing import List, Tuple
from pathlib import Path

from containers import ExperimentContainer
from .config import ExperimentConfig
from .runner import ExperimentRunner
from analysis.profiler import profiler


class ParallelExperimentManager:
    """
    Gerencia a execução de múltiplos experimentos em paralelo.
    """

    def __init__(self, configs: List[ExperimentConfig], filepaths: List[str], max_processes: int):
        self.configs = configs
        self.filepaths = filepaths
        self.max_processes = min(max_processes, cpu_count())
        self.experiment_container = ExperimentContainer()

    def run_all(self) -> List[dict]:
        """Executa todos os experimentos configurados usando um pool de processos."""
        print(f"Iniciando {len(self.configs)} experimentos em {self.max_processes} processos paralelos...")
        start_time = time.time()
        
        experiments = []
        for i, config in enumerate(self.configs):
            # A configuração e criação do runner acontece aqui, antes de ser enviado para o pool
            config_dict = asdict(config)
            self.experiment_container.config.from_dict(config_dict)
            runner = self.experiment_container.runner(test_filename=Path(self.filepaths[i]).name)
            experiments.append((runner, self.filepaths[i]))

        with Pool(self.max_processes) as pool:
            results = pool.starmap(run_experiment, experiments)

        total_duration = time.time() - start_time
        print(f"--- Fim de todos os experimentos | Duração Total: {total_duration:.2f}s ---")

        profiler.report()

        return results


def run_experiment(runner: ExperimentRunner, filepath: str) -> dict:
    """
    Função alvo para o pool de processos. Executa um único runner.
    """
    result = runner.run()
    result["filename"] = filepath
    
    # O relatório de cada processo filho será impresso ao final de sua execução
    profiler.report()
    
    return result