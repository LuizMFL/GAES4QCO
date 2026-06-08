import json
import warnings
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import time

import numpy as np
from qiskit.quantum_info import Statevector
from qiskit import transpile

# Ignora os avisos de deprecamento do Qiskit IBM Provider para limpar o terminal
warnings.filterwarnings("ignore", category=UserWarning, module="qiskit_ibm_provider.api.session")

from containers import AppContainer
from experiment.config import ExperimentConfig

# Constantes globais
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = PROJECT_ROOT / "tests"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_JSON = RESULTS_DIR / "final_benchmark_ranking.json"
SHOTS = 2 ** 15


def process_target(target_group_data: dict) -> dict:
    """
    Função executada em um processo isolado (Worker).
    Recebe os dados de um Circuito Alvo e a lista de testes associados a ele.
    Realiza a simulação de ruído e retorna o relatório local.
    """
    target_key = target_group_data["target_key"]
    target_path = target_group_data["target_path"]
    associated_tests = target_group_data["associated_tests"]

    print(f"[Worker] 🚀 Iniciando avaliação do Alvo: {target_key}")

    # Re-instancia o container dentro do processo filho,
    # pois objetos como BackendV2 não são 'pickláveis' (serializáveis) entre processos de forma segura.
    container = AppContainer()
    container.config.from_dict({
        "quantum": {
            "num_qubits": ExperimentConfig.num_qubits,
            "allowed_gates": None
        }
    })

    circuit_factory = container.circuit().circuit_factory()
    adapter = container.circuit().qiskit_adapter()
    error_analyzer = container.error_analyzer()
    backend = container.generic_backend()

    report_entry = {
        "target_name": target_key,
        "optimized_circuits": []
    }

    # 1. Avalia o Circuito Alvo Original
    if not target_path.exists():
        print(f"[Worker] ⚠️ Arquivo alvo não encontrado: {target_path}")
        return report_entry

    with open(target_path, "r", encoding="utf-8") as f:
        target_data = json.load(f)

    target_circuit = circuit_factory.create_from_dict(target_data)
    qiskit_target = adapter.from_domain(target_circuit)
    target_sv = Statevector.from_instruction(qiskit_target)

    target_error = error_analyzer.calculate_error_rate(
        circuit=target_circuit,
        target_statevector=target_sv,
        shots=SHOTS,
        verbose=False
    )

    transpiled_target = transpile(qiskit_target, backend=backend, optimization_level=0)

    report_entry["target_error"] = target_error
    report_entry["target_logical_depth"] = target_circuit.depth
    report_entry["target_physical_depth"] = transpiled_target.depth()

    # 2. Avalia todos os circuitos otimizados deste alvo
    for test in associated_tests:
        test_name = test["test_name"]
        circuits_dir = test["circuits_dir"]

        circuit_files = list(Path(circuits_dir).glob("*.json"))

        for circ_file in circuit_files:
            with open(circ_file, "r", encoding="utf-8") as f:
                circ_data = json.load(f)

            opt_circuit = circuit_factory.create_from_dict(circ_data)

            try:
                opt_error = error_analyzer.calculate_error_rate(
                    circuit=opt_circuit,
                    target_statevector=target_sv,
                    shots=SHOTS,
                    verbose=False
                )

                qiskit_opt = adapter.from_domain(opt_circuit)
                transpiled_opt = transpile(qiskit_opt, backend=backend, optimization_level=0)

                report_entry["optimized_circuits"].append({
                    "test_name": test_name,
                    "filename": circ_file.name,
                    "error_rate": opt_error,
                    "logical_depth": opt_circuit.depth,
                    "physical_depth": transpiled_opt.depth(),
                    "fidelity_approx": opt_circuit.fidelity
                })
            except Exception as e:
                print(f"[Worker] ❌ Erro ao avaliar {circ_file.name} no alvo {target_key}: {e}")

    print(f"[Worker] ✅ Concluído alvo: {target_key} ({len(report_entry['optimized_circuits'])} circuitos avaliados)")
    return report_entry


def main():
    print("🚀 Mapeando arquitetura de testes para processamento paralelo...")

    test_files = list(TESTS_DIR.glob("*.json"))
    if not test_files:
        print("⚠️ Nenhum arquivo de teste encontrado.")
        return

    # === FASE 1: MAPEAMENTO DOS TAREFAS ===
    # Agrupa quais pastas de circuitos otimizados pertencem a qual alvo (Target).
    targets_map = defaultdict(lambda: {"target_path": None, "associated_tests": []})

    for test_file in test_files:
        test_name = test_file.stem
        with open(test_file, "r", encoding="utf-8") as f:
            test_config = json.load(f)

        target_rel_path = test_config.get("filename_target_circuit")
        if not target_rel_path:
            continue
        if not target_rel_path.endswith(".json"):
            target_rel_path += ".json"

        target_path = PROJECT_ROOT / target_rel_path
        target_key = target_path.name

        targets_map[target_key]["target_path"] = target_path

        phases = test_config.get("phases", [])
        if not phases:
            continue

        last_phase_path = phases[-1].get("result_filepath")
        if not last_phase_path:
            continue

        circuits_dir = PROJECT_ROOT / last_phase_path.replace("_results.json", "_circuits")
        if circuits_dir.exists():
            targets_map[target_key]["associated_tests"].append({
                "test_name": test_name,
                "circuits_dir": str(circuits_dir)
            })

    # Prepara os pacotes de dados brutos que serão enviados para os processos filhos.
    task_payloads = []
    for target_key, data in targets_map.items():
        if data["associated_tests"]:
            task_payloads.append({
                "target_key": target_key,
                "target_path": data["target_path"],
                "associated_tests": data["associated_tests"]
            })

    print(f"📊 Encontrados {len(task_payloads)} circuitos alvo distintos para avaliação.")

    # === FASE 2: EXECUÇÃO EM PARALELO ===
    start_time = time.time()
    max_workers = min(len(task_payloads), cpu_count())
    print(f"🧠 Distribuindo tarefas em {max_workers} núcleos de processamento...")

    # Cria o Pool e mapeia as tarefas. O Pool cuidará de instanciar os processos filhos.
    with Pool(processes=max_workers) as pool:
        results_list = pool.map(process_target, task_payloads)

    elapsed_time = time.time() - start_time
    print(f"⏱️ Simulação em lote concluída em {elapsed_time:.2f} segundos.")

    # Reagrupa os resultados no dicionário 'report'
    report = {}
    for res in results_list:
        if res and "target_error" in res:
            report[res["target_name"]] = res

    # === FASE 3: GERAÇÃO DO RANKING AGRUPADO ===
    print("\n" + "=" * 70)
    print("🏆 RANKING FINAL DE RESISTÊNCIA A RUÍDO NISQ (AGRUPADO) 🏆")
    print("=" * 70)

    for target_key, data in report.items():
        print(f"\n🎯 ALVO: {target_key}")
        print(f"   Profundidade (Lógica / Física): {data['target_logical_depth']} / {data['target_physical_depth']}")
        print(f"   Taxa de Erro Original (TVD):    {data['target_error']:.2%}")
        print("-" * 70)

        sorted_circuits = sorted(data["optimized_circuits"], key=lambda x: x["error_rate"])
        if not sorted_circuits:
            print("   Nenhum circuito otimizado avaliado para este alvo.")
            continue

        for idx, circ in enumerate(sorted_circuits, start=1):
            circ["global_rank"] = idx

        target_rank = 1
        for circ in sorted_circuits:
            if circ["error_rate"] < data["target_error"]:
                target_rank += 1
            else:
                break

        total_competitors = len(sorted_circuits) + 1
        rank_percentile = (target_rank / total_competitors) * 100
        print(f"📍 Posição do Circuito Alvo no Rank: #{target_rank} de {total_competitors} (Top {rank_percentile:.1f}%)")
        print("\n🏅 TOP 3 CIRCUITOS POR TESTE (Ordenado pelo melhor desempenho global):\n")

        grouped_by_test = defaultdict(list)
        for circ in sorted_circuits:
            grouped_by_test[circ["test_name"]].append(circ)

        sorted_tests = sorted(grouped_by_test.items(), key=lambda item: item[1][0]["global_rank"])

        for test_name, circs in sorted_tests:
            best_global = circs[0]["global_rank"]
            print(f"🧪 Teste: {test_name} (Melhor Rank: #{best_global})")

            top_3 = circs[:3]
            for i, circ in enumerate(top_3, start=1):
                err = circ["error_rate"]
                global_r = circ["global_rank"]
                l_depth = circ["logical_depth"]
                p_depth = circ["physical_depth"]
                fid = circ.get("fidelity_approx", 0.0)

                is_better = "🟢 VENCEU " if err < data["target_error"] else "🔴 PERDEU"

                print(
                    f"   [{is_better}] Global #{global_r:03d} | Erro: {err:.2%} | Fid: {fid:.4f} | Depths (Log/Fis): {l_depth:02d}/{p_depth:02d}")
                print(f"                  Arquivo: {circ['filename']}")
            print("")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"💾 Relatório agrupado salvo em: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
