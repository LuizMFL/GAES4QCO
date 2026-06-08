from typing import Optional
import logging
import time

from analysis.interfaces import IDistanceMetric
from evolutionary_algorithm.interfaces import ISelectionStrategy, IMutationPopulation, IPopulationCrossover
from evolutionary_algorithm.population import Population
from evolutionary_algorithm.rate_adapter import IRateAdapter
from .interfaces import IFitnessEvaluator, IProgressObserver, IFitnessShaper


class Optimizer:
    """
    O motor do Algoritmo Genético, implementando um modelo (μ + λ) com elitismo
    e mutação forçada para garantir a diversidade.
    """

    def __init__(
            self,
            fitness_evaluator: IFitnessEvaluator,
            parent_selection: ISelectionStrategy,
            survivor_selection: ISelectionStrategy,
            crossover: IPopulationCrossover,
            mutation: IMutationPopulation,
            rate_adapter: IRateAdapter,
            fitness_shaper: IFitnessShaper,
            observer: IProgressObserver
    ):
        self._fitness_evaluator = fitness_evaluator
        self._parent_selection = parent_selection
        self._survivor_selection = survivor_selection
        self._crossover = crossover
        self._mutation = mutation
        self._rate_adapter = rate_adapter
        self._fitness_shaper = fitness_shaper
        self._observer = observer
        self._total_evaluations = 0

    def run(
            self,
            initial_population: Population,
            max_generations: int,
            fidelity_threshold: Optional[float]
    ) -> Population:
        phase_start_time = time.time()
        self._total_evaluations = 0

        for ind in initial_population.get_individuals():
            ind.fitness = None
            ind.base_fitness = None
            ind.fidelity = None

        current_population = initial_population
        stopping_reason = "max_generations_reached"

        logging.info("Evaluating initial population...")
        self._evaluate_population(current_population)

        for gen in range(max_generations):
            mutation_rate = getattr(self._mutation, 'mutation_rate', 0)
            crossover_rate = getattr(self._crossover, 'crossover_rate', 0)
            if self._observer:
                self._observer.update(gen, current_population, mutation_rate, crossover_rate)

            # 1. Adaptação de Taxas
            current_diversity = current_population.calculate_structural_diversity()
            current_rates = self._rate_adapter.adapt(current_diversity)
            self._crossover.crossover_rate = current_rates.crossover_rate
            self._mutation.mutation_rate = current_rates.mutation_rate
            
            # 2. Seleção de Pais
            parent_population = self._parent_selection.select(current_population)
            
            # 3. Crossover (separa filhos de pais não modificados)
            offspring_from_crossover, unmodified_parents = self._crossover.run(parent_population)

            # 4. Mutação Forçada (100% de taxa) nos pais não cruzados
            forced_mutation_rate_backup = self._mutation.mutation_rate
            self._mutation.mutation_rate = 1.0
            mutated_clones = self._mutation.mutate(unmodified_parents)
            self._mutation.mutation_rate = forced_mutation_rate_backup

            # 5. Mutação Normal (taxa da geração) nos filhos do crossover
            mutated_offspring = self._mutation.mutate(offspring_from_crossover)

            # 6. Avalia apenas os novos indivíduos
            newly_created = Population(mutated_clones.get_individuals() + mutated_offspring.get_individuals())
            self._evaluate_population(newly_created)

            combined_population = Population(current_population.get_individuals() + newly_created.get_individuals())

            # 7. Aplica o shaper nos filhos antes da seleção
            self._fitness_shaper.shape(combined_population)

            # 8. Seleção de Sobreviventes: (μ + λ)
            current_population = self._survivor_selection.select(combined_population)

            if fidelity_threshold:
                best_ind = current_population.get_fittest()
                if best_ind and best_ind.fidelity is not None and best_ind.fidelity >= fidelity_threshold:
                    logging.info(f"-> Fidelity threshold {fidelity_threshold} reached at generation {gen}.")
                    stopping_reason = "fidelity_threshold_reached"
                    break
        
        final_gen_index = max_generations if stopping_reason == "max_generations_reached" else gen + 1
        phase_duration = time.time() - phase_start_time

        if self._observer:
            self._observer.update(final_gen_index, current_population, mutation_rate, crossover_rate)
            best_circuit = current_population.get_fittest()
            if best_circuit:
                self._observer.set_summary(
                    duration_seconds=phase_duration,
                    final_generation=final_gen_index,
                    total_evaluations=self._total_evaluations,
                    stopping_reason=stopping_reason,
                    best_circuit=best_circuit
                )
                self._observer.save()

        return current_population

    def _evaluate_population(self, population: Population):
        evaluated_count = 0
        for individual in population.get_individuals():
            if individual.fitness is not None:
                continue

            evaluated_count += 1
            individual.fitness, individual.fidelity = self._fitness_evaluator.evaluate(individual)
            individual.base_fitness = individual.fitness
            self._total_evaluations += 1
        
        if evaluated_count > 0:
            logging.debug(f"Evaluated {evaluated_count} new individuals.")
