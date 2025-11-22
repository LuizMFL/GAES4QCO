import json
from pathlib import Path
from typing import Dict, List
import numpy as np

from analysis.circuit_error_evaluator import CircuitErrorEvaluator
from analysis.loader import JsonDataLoader
from analysis.plotter import (
    EvolutionPlotter,
    FidelityDepthPlotter,
    ErrorRatePlotter, GroupErrorRatePlotter,
)
from containers import AppContainer
from experiment.runner import circuits_folder_path
from analysis.utils import dataclass_to_primitive
from experiment.config import ExperimentConfig


PROJECT_PATH = Path(__file__).parents[2]


# Campos ignorados na definição do “grupo-base”
IGNORED_FIELDS = {"seed", "seed_target", "filename_target_circuit"}


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def base_signature(config_dict: dict) -> dict:
    """
    Remove seed, seed_target e filename_target_circuit.
    O restante define a identidade da configuração.
    """
    return {k: v for k, v in config_dict.items() if k not in IGNORED_FIELDS}


def main():
    """
    Nova versão do analisador:
    - Agrupa experimentos pela pasta base gerada pelo ExperimentConfig.
    - Cada pasta contém diferentes seeds da mesma configuração.
    - Avaliação é feita ao nível do grupo (configuração geral).
    """

    container = AppContainer()
    container.config.from_dict({"num_qubits": ExperimentConfig.num_qubits})
    qc_container = container.circuit()

    results_root = PROJECT_PATH / "results"
    grouped_roots = [p for p in results_root.iterdir() if p.is_dir() and "pha=" in p.name]

    plots_dir = PROJECT_PATH / "results" / "grouped_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    json_loader = JsonDataLoader()
    evo_plotter = EvolutionPlotter()
    fid_plotter = FidelityDepthPlotter()
    error_plotter = GroupErrorRatePlotter()

    group_metrics = []

    print(f"Encontradas {len(grouped_roots)} pastas de configuração-base.")

    # ------------------------------------------------------------
    # PROCESSAMENTO DE CADA GRUPO
    # ------------------------------------------------------------
    for group_dir in grouped_roots:
        print(f"\n📁 Avaliando grupo: {group_dir.name}")

        # A pasta possui vários experimentos: uuid_config.json, uuid_results.json, uuid_circuits/
        exp_configs = sorted(group_dir.glob("*_config.json"))
        exp_results = sorted(group_dir.glob("*_results.json"))

        if not exp_configs or not exp_results:
            print("⚠️ Nenhum conjunto válido encontrado neste grupo.")
            continue

        # Todos os filhos possuem a mesma configuração-base (exceto seeds)
        first_cfg = load_json(exp_configs[0])
        base_cfg = base_signature(first_cfg)

        # Determinar K-elite da configuração original
        elitism_size = first_cfg.get("elitism_size", 1)

        # Instância do avaliador (usa target de cada seed!)
        evaluator_cache: Dict[str, CircuitErrorEvaluator] = {}

        # ------------------------------------------------------------
        # MÉTRICAS POR SEED
        # ------------------------------------------------------------
        seed_errors = []     # média da elite de cada seed
        seed_variances = []  # variância interna
        seed_best = []       # melhor indivíduo da seed

        for cfg_path, res_path in zip(exp_configs, exp_results):
            cfg = load_json(cfg_path)

            seed = cfg["seed"]
            filename_target = cfg["filename_target_circuit"]
            target_path = (PROJECT_PATH / filename_target).with_suffix(".json")

            # Cache para evitar recarregar o mesmo statevector
            if filename_target not in evaluator_cache:
                evaluator_cache[filename_target] = CircuitErrorEvaluator(
                    target_circuit_path=target_path,
                    circuit_factory=qc_container.circuit_factory(),
                    qiskit_adapter=qc_container.qiskit_adapter(),
                    error_analyzer=container.error_analyzer(),
                    shots=2 ** 10,
                    verbose=False,
                )

            evaluator = evaluator_cache[filename_target]

            # Descobrir a pasta dos circuitos desta seed
            circuits_folder = circuits_folder_path(Path(cfg_path))
            circuit_files = sorted(circuits_folder.glob("*.json"))
            if not circuit_files:
                print(f"⚠️ Nenhum circuito final encontrado em {circuits_folder}")
                continue

            print(f"  🔍 Seed {seed}: avaliando {len(circuit_files)} circuitos…")
            errs = []
            for c in circuit_files:
                try:
                    e = evaluator.evaluate_circuit(c)
                    errs.append(e)
                except Exception as err:
                    print(f"⚠️ Erro ao avaliar {c.name}: {err}")

            if not errs:
                continue

            errs = sorted(errs)
            elite = errs[:elitism_size]

            seed_errors.append(float(np.mean(elite)))
            seed_variances.append(float(np.var(errs)))
            seed_best.append(float(np.min(errs)))

        # ------------------------------------------------------------
        # AGREGAÇÃO FINAL DO GRUPO
        # ------------------------------------------------------------
        if not seed_errors:
            print("⚠️ Grupo sem resultados suficientes.")
            continue

        aggregated = {
            "group_name": group_dir.name,
            "mean_elite_seed": float(np.mean(seed_errors)),
            "std_elite_seed": float(np.std(seed_errors)),
            "mean_variance_across_seeds": float(np.mean(seed_variances)),
            "best_individual_overall": float(np.min(seed_best)),
            "num_seeds": len(seed_errors),
            "elitism_size": elitism_size,
        }

        print(f"  ✅ Grupo {group_dir.name}: média elite-seed = {aggregated['mean_elite_seed']:.6f}")
        group_metrics.append(aggregated)

    # ------------------------------------------------------------
    #     RANKING ENTRE CONFIGURAÇÕES
    # ------------------------------------------------------------

    if not group_metrics:
        print("⚠️ Nenhuma métrica agregada encontrada.")
        return

    # Ordenação pela métrica principal: média da elite por seed
    group_metrics.sort(key=lambda g: g["mean_elite_seed"])

    print("\n🏆 TOP CONFIGURAÇÕES GERAIS (Fase 2)")
    for i, g in enumerate(group_metrics[:5], 1):
        print(f"#{i} | {g['group_name']} | elite-seed = {g['mean_elite_seed']:.6e}")

    # Salvar JSON
    ranking_path = plots_dir / "ranking_groups_fase2.json"
    with open(ranking_path, "w", encoding="utf-8") as f:
        json.dump(group_metrics, f, indent=4)

    print(f"💾 Ranking salvo em {ranking_path}")

    # Plot comparativo
    error_plot_path = plots_dir / "error_rate_comparison_groups.png"
    error_plotter.plot(group_metrics, str(error_plot_path), top_only=False)

    print("\n✅ Análise agrupada concluída com sucesso!")


if __name__ == "__main__":
    main()
