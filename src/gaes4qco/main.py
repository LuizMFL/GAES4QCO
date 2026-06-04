import sys
import logging
from pathlib import Path
from multiprocessing import cpu_count

from experiment.parallel_manager import ParallelExperimentManager
from experiment.test_loader import TestConfigLoader


def main():
    """
    Ponto de entrada para executar todos os experimentos definidos em `tests/`.
    """
    project_root = Path(__file__).resolve().parents[2]
    log_file_path = project_root / "execution.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    tests_dir = project_root / "tests"

    logging.info("🚀 Starting Quantum Circuit Evolution Experiments")
    logging.info(f"🔍 Loading test configurations from: {tests_dir}")

    loader = TestConfigLoader(tests_dir)
    experiment_configs, filepaths = loader.load_all(update_json=True)

    if not experiment_configs:
        logging.warning("⚠️ No valid test configurations found. Exiting.")
        sys.exit(0)

    num_processes = min(len(experiment_configs), cpu_count())
    logging.info(f"🧠 Running {len(experiment_configs)} experiments in parallel across {num_processes} processes...")
    
    manager = ParallelExperimentManager(
        configs=experiment_configs,
        filepaths=filepaths,
        max_processes=num_processes
    )

    all_results = manager.run_all()

    logging.info("\n=== 🧩 EXPERIMENT SUMMARY ===")
    for result in all_results:
        fname = result.get("filename", "unknown.json")
        seed = result.get("seed", "N/A")
        best_fit = result.get("best_fitness", 0.0)
        duration = result.get("duration_seconds", 0.0)
        logging.info(f"📄 {Path(fname).name}: Seed {seed} | Best Fitness = {best_fit:.6f} | Duration = {duration:.2f}s")

    if all_results:
        best_run = max(all_results, key=lambda r: r.get("best_fitness", 0.0))
        logging.info("\n🏆 BEST RUN SUMMARY")
        logging.info(f"📄 {Path(best_run['filename']).name} | Seed {best_run['seed']} | Fitness {best_run['best_fitness']:.6f}")

    logging.info("\n✅ All experiments completed successfully.")
    logging.info(f"Full log saved to {log_file_path}")


if __name__ == "__main__":
    main()