import glob
import json
import random
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
from qiskit.quantum_info import Statevector

from analysis.circuit_error_evaluator import CircuitErrorEvaluator
from analysis.loader import JsonDataLoader
from analysis.plotter import EvolutionPlotter, FidelityDepthPlotter, EvolutionAggregatedPlotter, \
    FidelityDepthAggregatedPlotter
from containers import AppContainer, QuantumCircuitContainer
from evolutionary_algorithm.selection import SelectionType
from experiment.config import ExperimentConfig, PhaseConfig
from experiment.runner import circuits_folder_path
from quantum_circuit.circuit import Circuit
from quantum_circuit.interfaces import IQuantumCircuitAdapter

PROJECT_PATH = Path(__file__).parents[2]


def main():
    """
    Carrega os dados de 'results/concatenated' e gera os gráficos individuais
    e agregados de cada conjunto de experimentos.

    Cada gráfico individual inclui um box com a configuração utilizada (por phase),
    lida a partir do arquivo de configuração do teste original.

    Todos os gráficos são salvos em 'results/concatenated/plots/'.
    """
    container = AppContainer()
    container.config.from_dict({"num_qubits": ExperimentConfig.num_qubits})
    quantum_circuit_container = container.circuit()

    concatenated_dir = PROJECT_PATH / "results" / "concatenated"
    plots_dir = concatenated_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not concatenated_dir.exists():
        print("❌ Diretório 'results/concatenated' não encontrado. Execute o concatenator primeiro.")
        return

    # Localiza todos os arquivos concatenados (ex: test_01_basic_concatenated_result.json)
    result_files = glob.glob(str(concatenated_dir / "*_concatenated_result.json"))

    if not result_files:
        print("⚠️ Nenhum arquivo concatenado encontrado em results/concatenated.")
        return

    # Instancia os componentes de análise
    loader = JsonDataLoader()
    single_plotter = EvolutionPlotter()
    fidelity_plotter = FidelityDepthPlotter()
    aggregated_evo = EvolutionAggregatedPlotter()
    aggregated_fid = FidelityDepthAggregatedPlotter()

    selection_combinations = {}
    min_depth, max_depth = 9999999, 0
    experiment_metrics = []  # ← armazenará erro médio final por configuração

    for filepath in result_files:
        try:
            result_data = loader.load(filepath)

            # === Carrega a configuração do teste original ===
            # O arquivo de teste original está em tests/, com o mesmo nome-base
            test_filename = (
                Path(filepath)
                .name.replace("_concatenated_result.json", ".json")
            )
            test_config_path = PROJECT_PATH / "tests" / test_filename

            if test_config_path.exists():
                with open(test_config_path, "r", encoding="utf-8") as f:
                    config_info = json.load(f)
                    json_data = config_info.copy()
                    experiment_config = _build_experiment(json_data)
            else:
                print(f"⚠️ Arquivo de configuração não encontrado: {test_config_path.name}")
                continue

            target_circuit_path = Path(__file__).parents[2] / config_info.get("filename_target_circuit")
            evaluator = CircuitErrorEvaluator(
                target_circuit_path=target_circuit_path,
                circuit_factory=quantum_circuit_container.circuit_factory(),
                qiskit_adapter=quantum_circuit_container.qiskit_adapter(),
                error_analyzer=container.error_analyzer(),
                shots=2 ** 20,
            )
            final_circuits_folder = [circuits_folder_path(Path(file_path)) for file_path in experiment_config.config_file_path]
            print(final_circuits_folder)
            if not final_circuits_folder[0].exists():
                print(f"⚠️ Pasta de circuitos finais não encontrada: {final_circuits_folder}")
                continue

            circuit_files = sorted(final_circuits_folder[0].glob("*.json"))
            if not circuit_files:
                print(f"⚠️ Nenhum circuito final encontrado em {final_circuits_folder}")
                continue

            print(f"🔍 Avaliando {len(circuit_files)} circuitos finais de '{test_filename}'...")
            errors_last_gen = []
            # === Avaliação de erro dos circuitos finais ===
            for circuit_path in circuit_files:
                try:
                    error_rate = evaluator.evaluate_circuit(circuit_path)
                    errors_last_gen.append(error_rate)
                except Exception as err:
                    print(f"⚠️ Falha ao avaliar circuito {circuit_path.name}: {err}")

            if errors_last_gen:
                mean_error = float(np.mean(errors_last_gen))
                experiment_metrics.append({
                    "name": Path(filepath).stem,
                    "config_path": test_config_path,
                    "mean_error": mean_error,
                    "num_circuits": len(errors_last_gen)
                })
                print(f"✅ {Path(filepath).stem} — Erro médio final: {mean_error:.6e}")
            else:
                print(f"⚠️ Nenhum erro válido para {Path(filepath).stem}")
            print("Passou")
            # === Nome base do arquivo (sem extensão) ===
            name = Path(filepath).stem

            # === Gráficos individuais ===
            output_filename = plots_dir / f"{name}.png"
            single_plotter.plot(result_data, str(output_filename), config_info=config_info)

            fidelity_output = plots_dir / f"{name}_fidelity_depth.png"
            fidelity_plotter.plot(result_data, str(fidelity_output), config_info=config_info)

            if config_info.get("phases"):
                first_phase = config_info["phases"][0]
                parent_sel = first_phase.get("parent_selection", "UNKNOWN").upper()
                survivor_sel = first_phase.get("survivor_selection", "UNKNOWN").upper()
                key = f"{parent_sel}_{survivor_sel}"
                selection_combinations.setdefault(key, []).append(result_data)
            min_depth = min(min_depth, config_info["min_depth"])
            max_depth = max(max_depth, config_info["max_depth"])
        except Exception as e:
            print(f"⚠️ Falha ao processar {filepath}: {e}")

    # === Seleção dos 5% melhores experimentos ===
    if experiment_metrics:
        experiment_metrics.sort(key=lambda x: x["mean_error"])
        top_k = max(1, int(len(experiment_metrics) * 0.05))
        best_configs = experiment_metrics[:top_k]

        print("\n🏆 Top 5% melhores configurações (menor erro médio):")
        for rank, cfg in enumerate(best_configs, start=1):
            print(f"#{rank}: {cfg['name']} | Erro médio = {cfg['mean_error']:.6e} | {cfg['config_path'].name}")

        ranking_path = plots_dir / "top_5_percent_configs.json"
        with open(ranking_path, "w", encoding="utf-8") as f:
            json.dump(best_configs, f, indent=4)
        print(f"💾 Ranking salvo em {ranking_path}")
    else:
        print("⚠️ Nenhuma métrica de erro foi calculada — ranking não gerado.")

    for combo, data_list in selection_combinations.items():
        if len(data_list) < 2:
            print(f"⚠️ Apenas um teste para {combo}, ignorando.")
            continue

        evo_output = plots_dir / f"aggregated_evolution_{combo}.png"
        fid_output = plots_dir / f"aggregated_fidelity_depth_{combo}.png"
        aggregated_evo.plot(data_list, str(evo_output))
        aggregated_fid.plot(data_list, str(fid_output), config_info={"max_depth": max_depth})
        print(f"📈 Gráficos agregados salvos para {combo}")

    print("\n✅ Todos os gráficos foram gerados com sucesso em 'results/concatenated/plots/'.")


def _build_phase(phase_dict: dict) -> PhaseConfig:
    """Constructs a PhaseConfig from its dict representation."""
    return PhaseConfig(
        use_stepsize=phase_dict["use_stepsize"],
        use_weighted_fitness=phase_dict["use_weighted_fitness"],
        use_adaptive_rates=phase_dict["use_adaptive_rates"],
        use_bandit_mutation=phase_dict["use_bandit_mutation"],
        parent_selection=SelectionType[phase_dict["parent_selection"].upper()],
        survivor_selection=SelectionType[phase_dict["survivor_selection"].upper()],
        use_fitness_sharing=phase_dict["use_fitness_sharing"],
        crossover_strategy=phase_dict["crossover_strategy"].lower(),
        generations=int(phase_dict["generations"]),
        fidelity_threshold_stop=phase_dict.get("fidelity_threshold_stop"),
    )


def _build_experiment(cfg: dict) -> ExperimentConfig:
    """Constructs an ExperimentConfig, filling only required fields."""
    phases = [_build_phase(p) for p in cfg["phases"]]
    # --- Campos obrigatórios ---
    target_sv, target_filepath = _load_or_create_target(
        num_qubits=cfg.get("num_qubits", ExperimentConfig.num_qubits),
        depth=cfg["target_depth"],
        seed_target=cfg["seed_target"],
        allowed_gates=cfg.get("allowed_gates"),
    )

    required_kwargs = dict(
        seed=cfg["seed"],
        max_depth=cfg["max_depth"],
        min_depth=cfg["min_depth"],
        target_depth=cfg["target_depth"],
        target_statevector_data=target_sv,
        filename_target_circuit=target_filepath,
        phases=phases,
        resume_from_checkpoint=cfg["resume_from_checkpoint"],
    )

    # --- Campos opcionais (se existirem, sobrescrevem os defaults do dataclass) ---
    optional_keys = [
        "allowed_gates", "num_qubits", "elitism_size", "population_size",
        "tournament_size", "crossover_rate", "mutation_rate",
        "min_mutation_rate", "max_mutation_rate",
        "min_crossover_rate", "max_crossover_rate",
        "diversity_threshold", "injection_rate",
        "sharing_radius", "alpha", "c_factor"
    ]
    for key in optional_keys:
        if key in cfg:
            required_kwargs[key] = cfg[key]

    return ExperimentConfig(**required_kwargs)


def _load_or_create_target(
        num_qubits: int, depth: int, seed_target: int, allowed_gates: Optional[List[str]]
) -> Tuple[Statevector, str]:
    """
    Loads an existing target circuit if available, or generates and saves a new one.
    Returns (statevector, filepath_base).
    """
    filepath_base = Path(__file__).parents[2] / f"target_seed_{seed_target}"
    filepath_json = Path(f"{filepath_base}.json")

    if filepath_json.exists():
        print(f"📂 Loading existing target circuit from {filepath_json.name}")
        with open(filepath_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        container = QuantumCircuitContainer()
        adapter = container.qiskit_adapter()
        factory = container.circuit_factory()
        circuit = factory.create_from_dict(data)
        qiskit_circuit = adapter.from_domain(circuit)
        target_sv = Statevector.from_instruction(qiskit_circuit)
        return target_sv, str(filepath_base)

    # Otherwise: generate deterministically
    print(f"⚙️ Generating new target circuit (seed={seed_target})...")
    random.seed(seed_target)
    np.random.seed(seed_target)

    container = QuantumCircuitContainer()
    container.config.from_dict({"quantum": {"allowed_gates": allowed_gates}})

    factory = container.circuit_factory()
    adapter = container.qiskit_adapter()

    domain_circuit = factory.create_random_circuit(
        num_qubits=num_qubits,
        max_depth=depth,
        min_depth=depth,
        use_evolutionary_strategy=False
    )

    # Save it
    _save_circuit_details(domain_circuit, adapter, str(filepath_base))

    qiskit_circuit = adapter.from_domain(domain_circuit)
    target_sv = Statevector.from_instruction(qiskit_circuit)
    return target_sv, str(filepath_base)


def _save_circuit_details(circuit: Circuit, adapter: IQuantumCircuitAdapter, filepath_base: str):
        """Saves both JSON representation and ASCII diagram of a circuit."""
        Path(filepath_base).parent.mkdir(parents=True, exist_ok=True)
        with open(f"{filepath_base}.json", "w", encoding="utf-8") as f:
            json.dump(circuit.to_dict(), f, indent=4)
        qiskit_circuit = adapter.from_domain(circuit)
        with open(f"{filepath_base}.txt", "w", encoding="utf-8") as f:
            f.write(str(qiskit_circuit.draw("text")))


if __name__ == "__main__":
    main()
