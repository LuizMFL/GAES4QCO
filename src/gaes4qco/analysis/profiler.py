import time
from functools import wraps
from collections import defaultdict
import numpy as np
import logging

class TimeProfiler:
    """
    Um singleton para coletar e reportar tempos de execução de funções.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TimeProfiler, cls).__new__(cls)
            cls._instance.timings = defaultdict(list)
        return cls._instance

    def add(self, func_name: str, elapsed_time: float):
        """Adiciona uma medição de tempo para uma função."""
        self.timings[func_name].append(elapsed_time)

    def report(self):
        """Imprime um relatório com as estatísticas de tempo de execução."""
        if not self.timings:
            return # Não imprime nada se nenhum dado foi coletado

        # Usa o logging para ser compatível com multiprocessing
        logging.info("\n--- ⏱️ Execution Time Profiling Report ---")
        
        sorted_funcs = sorted(self.timings.items(), key=lambda item: np.mean(item[1]), reverse=True)

        for func_name, times in sorted_funcs:
            mean_time = np.mean(times)
            total_time = np.sum(times)
            num_calls = len(times)
            std_dev = np.std(times)
            
            report_str = (
                f"• Function: {func_name}\n"
                f"  - Avg Time:   {mean_time:.6f}s\n"
                f"  - Total Time: {total_time:.4f}s\n"
                f"  - Num Calls:  {num_calls}\n"
                f"  - Std Dev:    {std_dev:.6f}s"
            )
            logging.info(report_str)
        logging.info("------------------------------------------")

# Instância global do profiler
profiler = TimeProfiler()

def profile_time(func):
    """
    Decorator que mede o tempo de execução de uma função e o registra no profiler.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        func_name = f"{func.__qualname__}"
        profiler.add(func_name, elapsed_time)
        
        return result
    return wrapper