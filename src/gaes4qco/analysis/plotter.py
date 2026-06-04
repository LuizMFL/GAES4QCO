from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np

from .interfaces import IPlotter
from .data_models import ResultData


def _clip(values):
    """Ensure all values are within [0, 1]."""
    return np.clip(values, 0.0, 1.0)


class EvolutionPlotter(IPlotter):
    """Generates a detailed evolution plot with clear phase markers and info boxes."""

    def plot(self, data: ResultData, output_path: str, config_info: Dict = None):
        print(f"Gerando gráfico em {output_path}...")

        generations = range(data.generation_count)

        # === Fitness ===
        avg_fit = _clip(np.array(data.average_fitness_per_generation))
        std_fit = np.array(data.std_dev_fitness_per_generation)
        best_fit = _clip(np.array(data.best_fitness_per_generation))

        lower_fit = _clip(avg_fit - std_fit)
        upper_fit = _clip(avg_fit + std_fit)

        # === Diversity ===
        diversity = _clip(np.array(data.structural_diversity_per_generation))

        # === Layout com área inferior extra ===
        fig = plt.figure(figsize=(14, 9))
        gs = fig.add_gridspec(2, 1, height_ratios=[3.5, 1])
        ax1 = fig.add_subplot(gs[0])

        # --- Eixo primário: Fitness ---
        color = 'tab:blue'
        ax1.set_xlabel('Geração')
        ax1.set_ylabel('Fitness', color=color)
        ax1.plot(generations, best_fit, color=color, linestyle='-', label='Melhor Fitness')
        ax1.plot(generations, avg_fit, color=color, linestyle='--', label='Fitness Médio')
        ax1.fill_between(generations, lower_fit, upper_fit, alpha=0.25, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, which='both', linestyle=':', linewidth=0.5)
        ax1.set_ylim(0, 1)
        ax1.set_xlim(0, data.generation_count)

        # --- Eixo secundário: Diversidade ---
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Diversidade Estrutural', color=color)
        ax2.plot(generations, diversity, color=color, linestyle='-.', label='Diversidade')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 1)
        ax2.set_xlim(0, data.generation_count)

        # --- Marcação visual das fases ---
        if config_info and "phases" in config_info:
            phases = config_info["phases"]
            phase_colors = plt.cm.Set2(np.linspace(0, 1, len(phases)))

            current_start = 0
            for i, (phase, color_box) in enumerate(zip(phases, phase_colors)):
                phase_len = phase.get("generations", 0)
                start_gen = current_start

                # Somente da segunda fase em diante: linha vertical marcando início
                if i > 0:
                    ax1.axvline(
                        x=start_gen,
                        color=color_box,
                        linestyle='--',
                        linewidth=1.3,
                        alpha=0.7,
                    )

                # Numeração da geração de início da phase
                ax1.text(
                    start_gen,
                    -0.08,  # abaixo do eixo X
                    str(start_gen),
                    color=color_box,
                    fontsize=9,
                    fontweight='bold',
                    ha='center',
                    va='top',
                    transform=ax1.get_xaxis_transform()
                )

                current_start += phase_len

        # --- Legenda combinada ---
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')

        plt.suptitle('Evolução do Fitness e Diversidade Genética por Geração', fontsize=13, y=0.94)

        # === Área inferior: Detalhes das Phases ===
        ax_config = fig.add_subplot(gs[1])
        ax_config.axis('off')

        if config_info and "phases" in config_info:
            phases = config_info["phases"]
            num_phases = len(phases)
            phase_colors = plt.cm.tab10(np.linspace(0, 1, num_phases))

            x_positions = np.linspace(0.02, 0.9, num_phases)
            for i, (phase, color_box) in enumerate(zip(phases, phase_colors)):
                true_flags = []
                if phase.get("use_stepsize"): true_flags.append("StepSize")
                if phase.get("use_adaptive_rates"): true_flags.append("AdaptiveRates")
                if phase.get("use_weighted_fitness"): true_flags.append("WeightedFitness")
                if phase.get("use_fitness_sharing"): true_flags.append("FitnessSharing")
                if phase.get("use_bandit_mutation"): true_flags.append("BanditMutation")

                text_lines = [
                    f"Phase {i + 1}",
                    f"Gerações: {phase.get('generations', '-')}",
                    f"ParentSel: {phase.get('parent_selection', '-')}",
                    f"SurvivorSel: {phase.get('survivor_selection', '-')}",
                    f"Crossover: {phase.get('crossover_strategy', '-')}"
                ]
                if true_flags:
                    text_lines.append("Ativos: " + ", ".join(true_flags))

                ax_config.text(
                    x_positions[i],
                    0.5,
                    "\n".join(text_lines),
                    fontsize=9,
                    va='center',
                    ha='left',
                    color='black',
                    bbox=dict(
                        boxstyle="round,pad=0.4",
                        facecolor=(color_box[0], color_box[1], color_box[2], 0.2),
                        edgecolor=color_box,
                        linewidth=1.2,
                    ),
                    transform=ax_config.transAxes
                )

        plt.subplots_adjust(hspace=0.35)
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
        plt.close()
        print("✅ Gráfico salvo com sucesso.")


class EvolutionAggregatedPlotter:
    """Gera um gráfico com a média e desvio padrão de múltiplas execuções."""

    def plot(self, all_results: List[ResultData], output_path: str):
        print(f"Gerando gráfico agregado em {output_path}...")

        # Número mínimo de gerações entre execuções
        min_generations = min(res.generation_count for res in all_results)
        generations = range(min_generations)

        # Matrizes 2D (cada linha: geração, cada coluna: execução)
        best_fitness_matrix = np.array(
            [_clip(res.best_fitness_per_generation[:min_generations]) for res in all_results]
        )
        avg_fitness_matrix = np.array(
            [_clip(res.average_fitness_per_generation[:min_generations]) for res in all_results]
        )
        diversity_matrix = np.array(
            [_clip(res.structural_diversity_per_generation[:min_generations]) for res in all_results]
        )

        # === Estatísticas ===
        mean_best_fitness = _clip(np.mean(best_fitness_matrix, axis=0))
        std_best_fitness = np.std(best_fitness_matrix, axis=0)
        mean_avg_fitness = _clip(np.mean(avg_fitness_matrix, axis=0))
        std_avg_fitness = np.std(avg_fitness_matrix, axis=0)
        mean_diversity = _clip(np.mean(diversity_matrix, axis=0))
        std_diversity = np.std(diversity_matrix, axis=0)
        best_global_fitness = np.max(best_fitness_matrix, axis=0)

        # Preenchimentos limitados
        lower_best = _clip(mean_best_fitness - std_best_fitness)
        upper_best = _clip(mean_best_fitness + std_best_fitness)
        lower_avg = _clip(mean_avg_fitness - std_avg_fitness)
        upper_avg = _clip(mean_avg_fitness + std_avg_fitness)
        lower_div = _clip(mean_diversity - std_diversity)
        upper_div = _clip(mean_diversity + std_diversity)

        # === Criação da figura ===
        fig, ax1 = plt.subplots(figsize=(14, 7))

        # --- Fitness ---
        color = 'tab:blue'
        ax1.set_xlabel('Geração')
        ax1.set_ylabel('Fitness', color=color)
        ax1.grid(True, which='both', linestyle=':', linewidth=0.5)

        # Linha de média do melhor fitness
        ax1.plot(generations, mean_best_fitness, color=color, linestyle='-', label='Média do Melhor Fitness')
        ax1.fill_between(generations, lower_best, upper_best, alpha=0.2, color=color)

        # Linha de média do fitness médio
        ax1.plot(generations, mean_avg_fitness, color='cyan', linestyle='--', label='Média do Fitness Médio')
        ax1.fill_between(generations, lower_avg, upper_avg, alpha=0.15, color='cyan')

        # === NOVA LINHA: melhor fitness global ===
        ax1.plot(
            generations,
            best_global_fitness,
            color='black',
            linestyle='-',
            linewidth=2.2,
            label='Melhor Fitness Global (entre execuções)'
        )

        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 1)
        ax1.set_xlim(0, min_generations)

        # --- Diversidade ---
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Diversidade Estrutural Média', color=color)
        ax2.plot(generations, mean_diversity, color=color, linestyle='-.', label='Média Diversidade')
        ax2.fill_between(generations, lower_div, upper_div, alpha=0.2, color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim(0, 1)
        ax2.set_xlim(0, min_generations)

        # Legendas combinadas
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')

        plt.title(f'Desempenho Agregado de {len(all_results)} Execuções', pad=20)
        fig.tight_layout(rect=(0, 0, 1, 0.97))

        # === Salvamento ===
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
        plt.close()
        print("✅ Gráfico agregado salvo com sucesso.")


class FidelityDepthPlotter(IPlotter):
    """
    Generates a detailed evolution plot showing the behavior of fidelity and depth per generation.
    Includes best, average, and deviation bands for both metrics, and renders phase boxes below.
    """

    def plot(self, data: ResultData, output_path: str, config_info: Dict = None):
        print(f"Gerando gráfico de fidelidade e profundidade em {output_path}...")

        generations = range(data.generation_count)

        # === Fidelidade ===
        avg_fid = _clip(np.array(data.average_fidelity_per_generation))
        std_fid = np.array(data.std_dev_fidelity_per_generation)
        best_fid = _clip(np.array(data.best_fidelity_per_generation))

        lower_fid = _clip(avg_fid - std_fid)
        upper_fid = _clip(avg_fid + std_fid)

        # === Profundidade (pode estar ausente) ===
        avg_depth = np.array(data.average_depth_per_generation or np.zeros_like(avg_fid))
        std_depth = np.array(data.std_dev_depth_per_generation or np.zeros_like(std_fid))
        best_depth = np.array(data.max_depth_per_generation or np.zeros_like(avg_fid))

        # Acesso defensivo ao config_info
        min_d = config_info.get("min_depth", 0) if isinstance(config_info, dict) else 0
        max_d = config_info.get("max_depth", 40) if isinstance(config_info, dict) else 40

        lower_depth = np.maximum(min_d, avg_depth - std_depth)
        upper_depth = np.minimum(max_d, avg_depth + std_depth)

        # === Layout ===
        fig = plt.figure(figsize=(14, 9))
        gs = fig.add_gridspec(2, 1, height_ratios=[3.5, 1])
        ax1 = fig.add_subplot(gs[0])

        # === Fidelidade ===
        color_fid = '#1f77b4'  # azul científico (fitness-like)
        ax1.set_xlabel("Geração")
        ax1.set_ylabel("Fidelidade", color=color_fid)
        ax1.plot(generations, best_fid, color=color_fid, linestyle='-', label="Melhor Fidelidade")
        ax1.plot(generations, avg_fid, color=color_fid, linestyle='--', label="Fidelidade Média")
        ax1.fill_between(generations, lower_fid, upper_fid, alpha=0.25, color=color_fid)
        ax1.tick_params(axis='y', labelcolor=color_fid)
        ax1.grid(True, which='both', linestyle=":", linewidth=0.5)
        ax1.set_ylim(0, 1)
        ax1.set_xlim(0, data.generation_count)

        # === Profundidade (eixo secundário) ===
        ax2 = ax1.twinx()
        color_depth = '#9467bd'  # roxo sutil
        ax2.set_ylabel("Profundidade Estrutural", color=color_depth)
        ax2.plot(generations, best_depth, color=color_depth, linestyle='-', label="Maior Profundidade")
        ax2.plot(generations, avg_depth, color=color_depth, linestyle='--', label="Profundidade Média")
        ax2.fill_between(generations, lower_depth, upper_depth, alpha=0.25, color=color_depth)
        ax2.tick_params(axis='y', labelcolor=color_depth)
        ax2.grid(True, which='both', linestyle=":", linewidth=0.5)
        ax2.set_ylim(min_d, max_d)
        ax2.set_xlim(0, data.generation_count)

        # === Marcação das Phases ===
        if config_info and "phases" in config_info:
            phases = config_info["phases"]
            phase_colors = plt.cm.Set2(np.linspace(0, 1, len(phases)))

            current_start = 0
            for i, (phase, color_box) in enumerate(zip(phases, phase_colors)):
                phase_len = phase.get("generations", 0)
                start_gen = current_start

                # Linha vertical de início (exceto Phase 1)
                if i > 0:
                    ax1.axvline(
                        x=start_gen,
                        color=color_box,
                        linestyle='--',
                        linewidth=1.3,
                        alpha=0.7,
                    )

                # Numeração da geração de início
                ax1.text(
                    start_gen,
                    -0.08,
                    str(start_gen),
                    color=color_box,
                    fontsize=9,
                    fontweight='bold',
                    ha='center',
                    va='top',
                    transform=ax1.get_xaxis_transform()
                )

                current_start += phase_len

        # === Legenda ===
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="center right")

        plt.suptitle("Evolução da Fidelidade e Profundidade Estrutural por Geração", fontsize=13, y=0.94)

        # === Área inferior com configuração das Phases ===
        ax_config = fig.add_subplot(gs[1])
        ax_config.axis('off')

        if config_info and "phases" in config_info:
            phases = config_info["phases"]
            num_phases = len(phases)
            phase_colors = plt.cm.tab10(np.linspace(0, 1, num_phases))

            x_positions = np.linspace(0.02, 0.9, num_phases)
            for i, (phase, color_box) in enumerate(zip(phases, phase_colors)):
                true_flags = []
                if phase.get("use_stepsize"): true_flags.append("StepSize")
                if phase.get("use_adaptive_rates"): true_flags.append("AdaptiveRates")
                if phase.get("use_weighted_fitness"): true_flags.append("WeightedFitness")
                if phase.get("use_fitness_sharing"): true_flags.append("FitnessSharing")
                if phase.get("use_bandit_mutation"): true_flags.append("BanditMutation")

                text_lines = [
                    f"Phase {i + 1}",
                    f"Gerações: {phase.get('generations', '-')}",
                    f"ParentSel: {phase.get('parent_selection', '-')}",
                    f"SurvivorSel: {phase.get('survivor_selection', '-')}",
                    f"Crossover: {phase.get('crossover_strategy', '-')}"
                ]
                if true_flags:
                    text_lines.append("Ativos: " + ", ".join(true_flags))

                ax_config.text(
                    x_positions[i],
                    0.5,
                    "\n".join(text_lines),
                    fontsize=9,
                    va='center',
                    ha='left',
                    color='black',
                    bbox=dict(
                        boxstyle="round,pad=0.4",
                        facecolor=(color_box[0], color_box[1], color_box[2], 0.2),
                        edgecolor=color_box,
                        linewidth=1.2,
                    ),
                    transform=ax_config.transAxes
                )

        plt.subplots_adjust(hspace=0.35)
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
        plt.close()
        print("✅ Gráfico de Fidelidade/Profundidade salvo com sucesso.")


class FidelityDepthAggregatedPlotter:
    """Plots aggregated fidelity and structural depth across multiple runs."""

    def plot(self, all_results: List[ResultData], output_path: str, config_info: Dict = None):
        print(f"Gerando gráfico agregado de fidelidade/profundidade em {output_path}...")

        min_generations = min(res.generation_count for res in all_results)
        generations = range(min_generations)

        # --- Matrizes ---
        best_fid_matrix = np.array(
            [_clip(res.best_fidelity_per_generation[:min_generations]) for res in all_results]
        )
        avg_fid_matrix = np.array(
            [_clip(res.average_fidelity_per_generation[:min_generations]) for res in all_results]
        )
        depth_matrix = np.array([
            np.array(res.average_depth_per_generation[:min_generations]) if res.average_depth_per_generation else np.zeros(min_generations)
            for res in all_results
        ])

        # --- Estatísticas ---
        mean_best_fid = _clip(np.mean(best_fid_matrix, axis=0))
        std_best_fid = np.std(best_fid_matrix, axis=0)
        mean_avg_fid = _clip(np.mean(avg_fid_matrix, axis=0))
        std_avg_fid = np.std(avg_fid_matrix, axis=0)
        mean_depth = np.mean(depth_matrix, axis=0)
        std_depth = np.std(depth_matrix, axis=0)
        best_global_fid = np.max(best_fid_matrix, axis=0)

        # --- Faixas ---
        lower_best_fid = _clip(mean_best_fid - std_best_fid)
        upper_best_fid = _clip(mean_best_fid + std_best_fid)
        lower_avg_fid = _clip(mean_avg_fid - std_avg_fid)
        upper_avg_fid = _clip(mean_avg_fid + std_avg_fid)
        lower_depth = np.maximum(config_info.get("min_depth", 0), mean_depth - std_depth)
        upper_depth = np.minimum(config_info.get("max_depth", np.max(mean_depth)), mean_depth + std_depth)

        # --- Gráfico ---
        fig, ax1 = plt.subplots(figsize=(14, 7))

        # Fidelidade
        color_fid = '#1f77b4'
        ax1.set_xlabel('Geração')
        ax1.set_ylabel('Fidelidade', color=color_fid)
        ax1.grid(True, linestyle=':', linewidth=0.5)

        ax1.plot(generations, mean_best_fid, color=color_fid, linestyle='-', label='Média do Melhor Fidelidade')
        ax1.fill_between(generations, lower_best_fid, upper_best_fid, alpha=0.2, color=color_fid)

        ax1.plot(generations, mean_avg_fid, color='cyan', linestyle='--', label='Média da Fidelidade Média')
        ax1.fill_between(generations, lower_avg_fid, upper_avg_fid, alpha=0.15, color='cyan')

        ax1.plot(generations, best_global_fid, color='black', linewidth=2.2, label='Melhor Fidelidade Global')
        ax1.tick_params(axis='y', labelcolor=color_fid)
        ax1.set_ylim(0, 1)
        ax1.set_xlim(0, min_generations)

        # Profundidade (eixo secundário)
        color_depth = '#9467bd'
        ax2 = ax1.twinx()
        ax2.set_ylabel('Profundidade Estrutural', color=color_depth)
        ax2.plot(generations, mean_depth, color=color_depth, linestyle='-.', label='Média Profundidade')
        ax2.fill_between(generations, lower_depth, upper_depth, alpha=0.25, color=color_depth)
        ax2.tick_params(axis='y', labelcolor=color_depth)
        ax2.set_ylim(config_info.get("min_depth", 0), config_info.get("max_depth"))
        ax2.set_xlim(0, min_generations)
        # Legendas
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')

        plt.title(f'Evolução Agregada da Fidelidade e Profundidade ({len(all_results)} execuções)', pad=20)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        plt.savefig(output_path, bbox_inches='tight', dpi=200)
        plt.close()
        print("✅ Gráfico agregado de fidelidade/profundidade salvo com sucesso.")


class ErrorRatePlotter:
    """
    Plota a taxa de erro média final de cada experimento.
    Pode exibir todos os experimentos ou apenas o top-k.
    """

    def plot(self, experiment_metrics: List[Dict], output_path: str, top_only: bool = False):
        if not experiment_metrics:
            print("⚠️ Nenhuma métrica de erro fornecida para o plot.")
            return

        # Ordena por erro crescente
        experiment_metrics = sorted(experiment_metrics, key=lambda x: x["mean_error"])
        names = [exp["name"] for exp in experiment_metrics]
        errors = [exp["mean_error"] * 100 for exp in experiment_metrics]
        num_experiments = len(errors)

        # Determina os Top-5%
        top_k = max(1, int(num_experiments * 0.05))
        top_indices = list(range(top_k))

        # Se for gráfico apenas dos Top-5%, recorta
        if top_only:
            names = names[:top_k]
            errors = errors[:top_k]
            num_experiments = top_k
            top_indices = list(range(num_experiments))
            print(f"📊 Plotando apenas os Top 5% ({num_experiments} experimentos).")

        # === Gráfico ===
        fig, ax = plt.subplots(figsize=(12, max(5, num_experiments * 0.3)))

        y_pos = np.arange(num_experiments)
        colors = ["#2ca02c" if i in top_indices else "#888888" for i in range(num_experiments)]

        bars = ax.barh(y_pos, errors, color=colors, edgecolor="black", alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Taxa de Erro Média (%)", fontsize=11)
        ax.set_title(
            "Top 5% Experimentos (menor erro)" if top_only
            else "Comparativo de Taxa de Erro Média por Experimento",
            fontsize=13,
            pad=15
        )
        ax.grid(axis="x", linestyle=":", linewidth=0.5)

        for bar, err in zip(bars, errors):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{err:.2f}%", va="center", ha="left", fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", dpi=200)
        plt.close()
        print(f"✅ Gráfico salvo em {output_path}")


class GroupErrorRatePlotter:
    """
    Plota taxa de erro média por grupo (configuração-base).
    Usado na Fase 2 onde experimentos são agrupados.
    """

    def plot(self, group_metrics: List[Dict], output_path: str, top_only: bool = False):
        if not group_metrics:
            print("⚠️ Nenhuma métrica de erro agrupada fornecida para o plot.")
            return

        # Ordena por erro crescente (mean_elite_seed)
        group_metrics = sorted(group_metrics, key=lambda x: x["mean_elite_seed"])

        names = [g["group_name"] for g in group_metrics]
        errors = [g["mean_elite_seed"] * 100 for g in group_metrics]
        num_groups = len(errors)

        # Determina top 5%
        top_k = max(1, int(num_groups * 0.05))
        top_indices = list(range(top_k))

        if top_only:
            names = names[:top_k]
            errors = errors[:top_k]
            num_groups = top_k
            top_indices = list(range(num_groups))
            print(f"📊 Plotando apenas os Top 5% ({num_groups} grupos).")

        fig, ax = plt.subplots(figsize=(12, max(5, num_groups * 0.3)))

        y_pos = np.arange(num_groups)
        colors = ["#2ca02c" if i in top_indices else "#888888" for i in range(num_groups)]

        bars = ax.barh(y_pos, errors, color=colors, edgecolor="black", alpha=0.85)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Taxa de Erro Média da Elite (%)", fontsize=11)
        ax.set_title(
            "Top 5% Configurações (Fase 2)" if top_only
            else "Comparativo de Configurações (Fase 2)",
            fontsize=14,
            pad=15
        )
        ax.grid(axis="x", linestyle=":", linewidth=0.5)

        # valores numéricos ao lado das barras
        for bar, err in zip(bars, errors):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{err:.2f}%", va="center", ha="left", fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches="tight", dpi=200)
        plt.close()
        print(f"✅ Gráfico salvo em {output_path}")
