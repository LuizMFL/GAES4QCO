import time
from multiprocessing import Pool, cpu_count, current_process, RLock
from typing import List
from pathlib import Path
from dataclasses import asdict
from tqdm import tqdm

from containers import ExperimentContainer
from .config import ExperimentConfig


# from analysis.profiler import profiler # Omitido para não sujar o terminal

def _init_pool(tqdm_lock):
    """Garante que todos os processos usem o mesmo lock de terminal"""
    tqdm.set_lock(tqdm_lock)


def _unpack_and_run(args):
    """Wrapper para desenpacotar argumentos do imap_unordered"""
    return run_experiment_task(*args)


def run_experiment_task(config_dict: dict, filepath: str) -> dict:
    # Captura o ID do processo (ex: 1, 2, 3...). Ele será a linha da barra!
    worker_id = current_process()._identity[0] if current_process()._identity else 1

    experiment_container = ExperimentContainer()
    
    # 2. Configura o container com os dados específicos deste experimento.
    experiment_container.config.from_dict(config_dict)
    
    # 3. Cria o runner a partir do container recém-configurado.
    runner = experiment_container.runner(test_filename=Path(filepath).name)

    # 4. Executa o experimento.
    # Passa o ID da linha para o Runner
    result = runner.run(position_id=worker_id)
    result["filename"] = filepath
    
    # 5. Imprime o relatório de profiling deste processo.
    # profiler.report()
    
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
        print(f"🚀 Iniciando {len(self.configs)} experimentos em {self.max_processes} processos paralelos...\n")
        start_time = time.time()
        
        # Envia apenas dados primitivos (dicionários e strings) para os workers.
        tasks = [(asdict(config), filepath) for config, filepath in zip(self.configs, self.filepaths)]
        results = []

        tqdm_lock = RLock()
        with Pool(processes=self.max_processes, initializer=_init_pool, initargs=(tqdm_lock,)) as pool:
            # Barra Global na linha 0
            for res in tqdm(
                    pool.imap_unordered(_unpack_and_run, tasks),
                    total=len(tasks),
                    desc="📊 Progresso Geral",
                    position=0,
                    leave=True,
                    dynamic_ncols=True,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            ):
                results.append(res)

        # Quebra de linhas para o prompt voltar ao normal abaixo do dashboard
        print("\n" * (self.max_processes + 1))

        total_duration = time.time() - start_time
        print(f"✅ Fim de todos os experimentos | Duração Total: {total_duration:.2f}s")
        return results
