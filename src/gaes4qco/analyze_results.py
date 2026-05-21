import json

import numpy as np
from pathlib import Path

from analysis.circuit_error_evaluator import CircuitErrorEvaluator
from analysis.loader import JsonDataLoader
from analysis.plotter import (
    EvolutionPlotter,
    FidelityDepthPlotter,
    EvolutionAggregatedPlotter,
    FidelityDepthAggregatedPlotter, ErrorRatePlotter,
)
from analysis.utils import dataclass_to_primitive
from containers import AppContainer
from experiment.test_loader import TestConfigLoader
from experiment.config import ExperimentConfig


PROJECT_PATH = Path(__file__).parents[2]


def main():
    """
    Analisa todos os testes definidos em `tests/`, calcula a taxa de erro média
    dos circuitos finais (última geração) e gera gráficos e ranking das melhores configurações.
    """
    container = AppContainer()
    container.config.from_dict({"num_qubits": ExperimentConfig.num_qubits})
    quantum_circuit_container = container.circuit()

    # === Diretórios ===
    tests_dir = PROJECT_PATH / "tests"
    concatenated_dir = PROJECT_PATH / "results" / "concatenated"
    plots_dir = concatenated_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # === Carrega todos os testes ===
    loader = TestConfigLoader(tests_dir)
    experiment_configs, filenames = loader.load_all(update_json=False)
    if not experiment_configs:
        print("⚠️ Nenhum teste válido encontrado em 'tests/'.")
        return

    # === Plotters ===
    json_loader = JsonDataLoader()
    single_plotter = EvolutionPlotter()
    fidelity_plotter = FidelityDepthPlotter()
    aggregated_evo = EvolutionAggregatedPlotter()
    aggregated_fid = FidelityDepthAggregatedPlotter()
    error_rate_plotter = ErrorRatePlotter()

    # === Métricas ===
    experiment_metrics = []
    selection_combinations = {}
    min_depth, max_depth = 9999999, 0

    # === Loop por TESTES ===
    for config, test_filename in zip(experiment_configs, filenames):
        try:
            test_config_path = tests_dir / test_filename
            print(f"\n🧪 Analisando teste: {test_filename}")

            # Caminho do circuito alvo
            target_circuit_path = (PROJECT_PATH / config.filename_target_circuit).with_suffix(".json")
            evaluator = CircuitErrorEvaluator(
                target_circuit_path=target_circuit_path,
                circuit_factory=quantum_circuit_container.circuit_factory(),
                qiskit_adapter=quantum_circuit_container.qiskit_adapter(),
                error_analyzer=container.error_analyzer(),
                shots=2 ** 10,
                verbose=False
            )

            # === Localiza resultados concatenados ===
            result_file = concatenated_dir / f"{test_filename.replace('.json', '_concatenated_result.json')}"
            if not result_file.exists():
                print(f"⚠️ Arquivo de resultados não encontrado: {result_file.name}")
                continue

            result_data = json_loader.load(str(result_file))

            # === Determina a última fase do experimento ===
            if not config.phases or not config.phases[-1].result_filepath:
                print(f"⚠️ Configuração {test_filename} não contém caminhos de resultados na última fase.")
                continue

            # Usa o caminho da última fase para encontrar a pasta de circuitos
            last_phase_result_path = PROJECT_PATH / config.phases[-1].result_filepath
            final_circuits_folder = last_phase_result_path.parent / last_phase_result_path.name.replace("_results.json", "_circuits")

            if not final_circuits_folder.exists():
                print(f"⚠️ Pasta de circuitos finais não encontrada: {final_circuits_folder}")
                continue

            circuit_files = sorted(final_circuits_folder.glob("*.json"))
            if not circuit_files:
                print(f"⚠️ Nenhum circuito final encontrado em {final_circuits_folder}")
                continue

            print(f"🔍 Avaliando {len(circuit_files)} circuitos finais na última fase...")
            errors_last_gen = []
            for circuit_path in circuit_files:
                try:
                    err_rate = evaluator.evaluate_circuit(circuit_path)
                    errors_last_gen.append(err_rate)
                except Exception as e:
                    print(f"⚠️ Erro ao avaliar {circuit_path.name}: {e}")

            if errors_last_gen:
                errors_last_gen_sorted = sorted(errors_last_gen)
                elite_errors = errors_last_gen_sorted[:config.elitism_size]

                mean_error = float(np.mean(elite_errors))
                best_error = float(np.min(errors_last_gen_sorted))

                experiment_metrics.append({
                    "name": test_filename.replace(".json", ""),
                    "config_path": str(test_config_path),
                    "mean_error": mean_error,
                    "best_error": best_error,
                    "k_elite_size": config.elitism_size,
                    "num_circuits": len(errors_last_gen_sorted),
                })

                print(f"✅ Erro médio da elite (top {config.elitism_size}): {mean_error:.6e} | Melhor: {best_error:.6e}")
            else:
                print("⚠️ Nenhum erro calculado para este teste.")

            # === Gera gráficos ===
            plot_name = test_filename.replace(".json", "")
            config_dict = dataclass_to_primitive(config)
            single_plotter.plot(result_data, str(plots_dir / f"{plot_name}.png"), config_info=config_dict)
            fidelity_plotter.plot(result_data, str(plots_dir / f"{plot_name}_fidelity_depth.png"), config_info=config_dict)

            if config.phases:
                first_phase = config.phases[0]
                parent_sel = first_phase.parent_selection.value.upper()
                survivor_sel = first_phase.survivor_selection.value.upper()
                key = f"{parent_sel}_{survivor_sel}"
                selection_combinations.setdefault(key, []).append(result_data)

            min_depth = min(min_depth, config.min_depth)
            max_depth = max(max_depth, config.max_depth)

        except Exception as e:
            print(f"❌ Falha ao processar teste {test_filename}: {e}")

    # === Ranking Top 5% ===
    if experiment_metrics:
        experiment_metrics.sort(key=lambda x: x["mean_error"])
        top_k = max(1, int(len(experiment_metrics) * 0.05))
        best_configs = experiment_metrics[:top_k]

        print("\n🏆 Top 5% melhores configurações (menor erro médio):")
        for i, cfg in enumerate(best_configs, start=1):
            print(f"#{i}: {cfg['name']} | Erro médio = {cfg['mean_error']:.6e}")

        ranking_path = plots_dir / "top_5_percent_configs.json"
        with open(ranking_path, "w", encoding="utf-8") as f:
            json.dump(best_configs, f, indent=4)
        print(f"💾 Ranking salvo em {ranking_path}")

        # === Gráfico comparativo apenas dos Top 5% ===
        error_plot_path_top = plots_dir / "error_rate_comparison_top5.png"
        error_rate_plotter.plot(best_configs, str(error_plot_path_top), top_only=False)

        # === (Opcional) gráfico completo com todos os experimentos ===
        error_plot_path_all = plots_dir / "error_rate_comparison_all.png"
        error_rate_plotter.plot(experiment_metrics, str(error_plot_path_all), top_only=False)
    else:
        print("⚠️ Nenhuma métrica de erro calculada — ranking não gerado.")

    # === Gera gráficos agregados ===
    for combo, data_list in selection_combinations.items():
        if len(data_list) < 2:
            continue
        evo_output = plots_dir / f"aggregated_evolution_{combo}.png"
        fid_output = plots_dir / f"aggregated_fidelity_depth_{combo}.png"
        aggregated_evo.plot(data_list, str(evo_output))
        aggregated_fid.plot(data_list, str(fid_output), config_info={"max_depth": max_depth})

    print("\n✅ Análise concluída com sucesso!")


if __name__ == "__main__":
    main()
