import time
from multiprocessing import Pool, cpu_count
from typing import List
from pathlib import Path
from dataclasses import asdict

from containers import ExperimentContainer
from .config import ExperimentConfig
from analysis.profiler import profiler


def run_experiment_task(config_dict: dict, filepath: str) -> dict:
    """
    Função alvo para cada processo. Cria um ambiente totalmente isolado para executar o experimento.
    """
    # 1. Cada processo cria seu próprio container.
    experiment_container = ExperimentContainer()
    
    # 2. Configura o container com os dados específicos deste experimento.
    experiment_container.config.from_dict(config_dict)
    
    # 3. Cria o runner a partir do container recém-configurado.
    runner = experiment_container.runner(test_filename=Path(filepath).name)
    
    # 4. Executa o experimento.
    result = runner.run()
    result["filename"] = filepath
    
    # 5. Imprime o relatório de profiling deste processo.
    profiler.report()
    
    return result


class ParallelExperimentManager:
    """
    Gerencia a execução de múltiplos experimentos em paralelo.
    """
    def __init__(self, configs: List[ExperimentConfig], filepaths: List[str], max_processes: int):
        self.configs = configs
        self.filepaths = filepaths
        self.max_processes = min(max_processes, cpu_count())

    def run_all(self) -> List[dict]:
        """Prepara e executa todos os experimentos em um pool de processos."""
        print(f"Iniciando {len(self.configs)} experimentos em {self.max_processes} processos paralelos...")
        start_time = time.time()
        
        # Envia apenas dados primitivos (dicionários e strings) para os workers.
        tasks = [(asdict(config), filepath) for config, filepath in zip(self.configs, self.filepaths)]

        with Pool(processes=self.max_processes) as pool:
            results = pool.starmap(run_experiment_task, tasks)

        total_duration = time.time() - start_time
        print(f"--- Fim de todos os experimentos | Duração Total: {total_duration:.2f}s ---")

        profiler.report()
        return results