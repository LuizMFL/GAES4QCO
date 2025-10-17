from analysis.loader import JsonDataLoader
from analysis.plotter import EvolutionPlotter, AggregatePlotter, FidelityDepthPlotter
from pathlib import Path
import glob
import json


PROJECT_PATH = Path(__file__).parents[2]


def main():
    """
    Carrega os dados de 'results/concatenated' e gera os gráficos individuais
    e agregados de cada conjunto de experimentos.

    Cada gráfico individual inclui um box com a configuração utilizada (por phase),
    lida a partir do arquivo de configuração do teste original.

    Todos os gráficos são salvos em 'results/concatenated/plots/'.
    """
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
    aggregate_plotter = AggregatePlotter()
    fidelity_plotter = FidelityDepthPlotter()

    grouped_results = {}

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

            config_info = {}
            if test_config_path.exists():
                with open(test_config_path, "r", encoding="utf-8") as f:
                    config_info = json.load(f)
            else:
                print(f"⚠️ Arquivo de configuração não encontrado: {test_config_path.name}")

            # === Nome base do arquivo (sem extensão) ===
            name = Path(filepath).stem

            # === Define o caminho de saída do gráfico individual ===
            output_filename = plots_dir / f"{name}.png"

            # === Gera gráfico com informações de configuração ===
            single_plotter.plot(result_data, str(output_filename), config_info=config_info)

            fidelity_output = plots_dir / f"{name}_fidelity_depth.png"
            fidelity_plotter.plot(result_data, str(fidelity_output), config_info=config_info)

            # === Classifica o teste em um grupo (ex: stepsize, sharing, nsga, normal) ===
            lower_name = name.lower()
            if "stepsize" in lower_name:
                group_key = "stepsize"
            elif "sharing" in lower_name:
                group_key = "sharing"
            elif "nsga" in lower_name:
                group_key = "nsga"
            else:
                group_key = "normal"

            grouped_results.setdefault(group_key, []).append(result_data)

        except Exception as e:
            print(f"⚠️ Falha ao processar {filepath}: {e}")

    # === Gráficos agregados por grupo ===
    for group, data_list in grouped_results.items():
        if not data_list:
            continue

        group_plot_path = plots_dir / f"aggregated_{group}_performance.png"
        aggregate_plotter.plot(data_list, str(group_plot_path))
        print(f"📊 Gráfico agregado salvo para grupo '{group}' → {group_plot_path.name}")

    print("\n✅ Todos os gráficos foram gerados com sucesso em 'results/concatenated/plots/'.")


if __name__ == "__main__":
    main()
