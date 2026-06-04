import json
from pathlib import Path
from typing import Dict, List
import numpy as np
from enum import Enum

from analysis.circuit_error_evaluator import CircuitErrorEvaluator
from analysis.plotter import GroupErrorRatePlotter
from containers import AppContainer
from experiment.test_loader import TestConfigLoader
from experiment.config import ExperimentConfig

PROJECT_PATH = Path(__file__).parents[2]
RESULTS_DIR = PROJECT_PATH / "results"

# Campos a serem ignorados ao criar a assinatura de um grupo de configuração
IGNORED_FIELDS_FOR_GROUPING = {
    "seed", "seed_target", "filename_target_circuit", 
    "config_file_path", "phases", "target_statevector_data", "resume_from_checkpoint"
}

def get_group_signature(config: ExperimentConfig) -> str:
    """
    Cria uma assinatura única para uma configuração de experimento, ignorando as sementes e dados de execução.
    """
    from dataclasses import asdict
    
    d = asdict(config)
    
    # Remove campos voláteis ou específicos da semente
    for key in IGNORED_FIELDS_FOR_GROUPING:
        d.pop(key, None)
        
    # A assinatura das fases é baseada em suas configurações, não nos caminhos de resultado
    phase_signatures = []
    for phase in config.phases:
        phase_dict = asdict(phase)
        phase_dict.pop("result_filepath", None)
        # Converte Enums para seus valores de string dentro de cada fase
        for k, v in phase_dict.items():
            if isinstance(v, Enum):
                phase_dict[k] = v.value
        phase_signatures.append(tuple(sorted(phase_dict.items())))
        
    d["phases"] = tuple(phase_signatures)
    
    # Função auxiliar para o json.dumps lidar com qualquer Enum que possa ter sobrado
    def enum_serializer(obj):
        if isinstance(obj, Enum):
            return obj.value
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    # Converte para uma string canônica para usar como chave de dicionário
    return json.dumps(d, sort_keys=True, default=enum_serializer)


def main():
    """
    Analisa os resultados agrupando experimentos com a mesma configuração base (ignorando a semente).
    Calcula métricas agregadas por grupo para determinar a melhor configuração de hiperparâmetros.
    """
    container = AppContainer()
    container.config.from_dict({"num_qubits": ExperimentConfig.num_qubits})
    qc_container = container.circuit()

    tests_dir = PROJECT_PATH / "tests"
    plots_dir = RESULTS_DIR / "grouped_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Carregar e Agrupar todos os Testes ---
    loader = TestConfigLoader(tests_dir)
    all_configs, all_filenames = loader.load_all(update_json=False)

    groups: Dict[str, List[ExperimentConfig]] = {}
    for config in all_configs:
        signature = get_group_signature(config)
        groups.setdefault(signature, []).append(config)

    print(f"Encontrados {len(all_configs)} testes, agrupados em {len(groups)} configurações-base.")

    # --- 2. Processar cada Grupo ---
    group_metrics = []
    for signature, configs_in_group in groups.items():
        first_config = configs_in_group[0]
        
        # Gera um nome legível para o grupo a partir do nome do primeiro arquivo de teste
        group_name = Path(all_filenames[all_configs.index(first_config)]).stem
        
        print(f"\n📁 Avaliando grupo: {group_name} ({len(configs_in_group)} seeds)")

        elitism_size = first_config.elitism_size
        evaluator_cache: Dict[str, CircuitErrorEvaluator] = {}
        
        seed_errors, seed_variances, seed_bests = [], [], []

        # --- 3. Processar cada Seed dentro do Grupo ---
        for config in configs_in_group:
            target_path = (PROJECT_PATH / config.filename_target_circuit).with_suffix(".json")
            
            if config.filename_target_circuit not in evaluator_cache:
                evaluator_cache[config.filename_target_circuit] = CircuitErrorEvaluator(
                    target_circuit_path=target_path,
                    circuit_factory=qc_container.circuit_factory(),
                    qiskit_adapter=qc_container.qiskit_adapter(),
                    error_analyzer=container.error_analyzer(),
                    shots=2 ** 10,
                    verbose=False,
                )
            evaluator = evaluator_cache[config.filename_target_circuit]

            # Reconstrói o caminho para a pasta de circuitos da última fase
            folder_names = list(config.get_config_foldername())
            hashes = list(config.get_config_hash())
            
            current_path = RESULTS_DIR
            for i in range(len(config.phases)):
                current_path = current_path / folder_names[i]
            
            final_circuits_folder = current_path / f"{hashes[-1]}_circuits"

            if not final_circuits_folder.exists():
                print(f"  ⚠️ Seed {config.seed}: Pasta de circuitos finais não encontrada em {final_circuits_folder}")
                continue

            circuit_files = sorted(final_circuits_folder.glob("*.json"))
            if not circuit_files:
                continue

            print(f"  🔍 Seed {config.seed}: avaliando {len(circuit_files)} circuitos…")
            errors = [evaluator.evaluate_circuit(c) for c in circuit_files]
            
            if not errors:
                continue

            errors = sorted(errors)
            elite_errors = errors[:elitism_size]
            
            seed_errors.append(float(np.mean(elite_errors)))
            seed_variances.append(float(np.var(errors)))
            seed_bests.append(float(np.min(errors)))

        # --- 4. Agregar Métricas do Grupo ---
        if not seed_errors:
            print("  ⚠️ Grupo sem resultados suficientes para análise.")
            continue

        aggregated = {
            "group_name": group_name,
            "mean_elite_seed": float(np.mean(seed_errors)),
            "std_elite_seed": float(np.std(seed_errors)),
            "mean_variance_across_seeds": float(np.mean(seed_variances)),
            "best_individual_overall": float(np.min(seed_bests)),
            "num_seeds": len(seed_errors),
            "elitism_size": elitism_size,
        }
        group_metrics.append(aggregated)
        print(f"  ✅ Média do erro da elite no grupo: {aggregated['mean_elite_seed']:.6f}")

    # --- 5. Ranking e Plot Final ---
    if not group_metrics:
        print("\n⚠️ Nenhuma métrica de grupo calculada. Encerrando.")
        return

    group_metrics.sort(key=lambda g: g["mean_elite_seed"])

    print("\n🏆 TOP CONFIGURAÇÕES GERAIS")
    for i, g in enumerate(group_metrics[:5], 1):
        print(f"#{i} | {g['group_name']} | Erro médio da elite = {g['mean_elite_seed']:.6e}")

    ranking_path = plots_dir / "ranking_grupos.json"
    with open(ranking_path, "w", encoding="utf-8") as f:
        json.dump(group_metrics, f, indent=4)
    print(f"💾 Ranking de grupos salvo em {ranking_path}")

    error_plotter = GroupErrorRatePlotter()
    error_plot_path = plots_dir / "error_rate_comparison_groups.png"
    error_plotter.plot(group_metrics, str(error_plot_path))

    print("\n✅ Análise agrupada concluída com sucesso!")


if __name__ == "__main__":
    main()