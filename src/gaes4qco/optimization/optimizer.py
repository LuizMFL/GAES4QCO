from typing import List, Optional, Dict, Tuple
import logging
import time

from evolutionary_algorithm.population_factory import PopulationFactory
from evolutionary_algorithm.interfaces import ISelectionStrategy, IMutationPopulation, IPopulationCrossover
from evolutionary_algorithm.population import Population
from evolutionary_algorithm.rate_adapter import IRateAdapter
from .interfaces import IFitnessEvaluator, IProgressObserver, IFitnessShaper


class Optimizer:
    """
    O motor do Algoritmo Genético, implementando um modelo geracional com elitismo.
    """

    def __init__(
            self,
            fitness_evaluator: IFitnessEvaluator,
            parent_selection: ISelectionStrategy,
            crossover: IPopulationCrossover,
            mutation: IMutationPopulation,
            population_factory: PopulationFactory,
            rate_adapter: IRateAdapter,
            fitness_shaper: IFitnessShaper,
            observer: IProgressObserver,
            elitism_size: int,
            population_size: int,
            **kwargs 
    ):
        self._fitness_evaluator = fitness_evaluator
        self._parent_selection = parent_selection
        self._crossover = crossover
        self._mutation = mutation
        self._rate_adapter = rate_adapter
        self._fitness_shaper = fitness_shaper
        self._observer = observer
        self._elitism_size = elitism_size
        self._population_size = population_size
        self._total_evaluations = 0
        self._fitness_cache: Dict[Tuple, Tuple[float, float]] = {}

    def run(
            self,
            initial_population: Population,
            max_generations: int,
            fidelity_threshold: Optional[float]
    ) -> Population:
        """
        Executa o fluxo do algoritmo genético usando um modelo geracional com elitismo.
        """
        phase_start_time = time.time()
        self._total_evaluations = 0
        self._fitness_cache.clear()
        current_population = initial_population
        stopping_reason = "max_generations_reached"

        logging.info("Evaluating initial population...")
        self._evaluate_population(current_population)

        for gen in range(max_generations):
            if self._observer:
                self._observer.update(gen, current_population, getattr(self._mutation, 'mutation_rate', 0), getattr(self._crossover, 'crossover_rate', 0))

            # 1. Elitismo: Os melhores indivíduos são copiados diretamente para a próxima geração.
            current_population.get_individuals().sort(key=lambda ind: ind.fitness, reverse=True)
            elites = current_population.get_individuals()[:self._elitism_size]

            # 2. Geração de Filhos para preencher o resto da população
            num_offspring_to_create = self._population_size - len(elites)
            
            # Adapta taxas
            current_diversity = current_population.calculate_structural_diversity()
            current_rates = self._rate_adapter.adapt(current_diversity)
            self._crossover.crossover_rate = current_rates.crossover_rate
            self._mutation.mutation_rate = current_rates.mutation_rate
            
            # Loop de reprodução
            offspring = []
            while len(offspring) < num_offspring_to_create:
                # Seleciona pais
                parents = self._parent_selection.select(current_population, num_to_select=2).get_individuals()
                
                # Aplica Crossover
                children = self._crossover.run(Population(parents)).get_individuals()
                
                # Aplica Mutação
                mutated_children = self._mutation.mutate(Population(children)).get_individuals()
                
                offspring.extend(mutated_children)

            # Pega o número exato de filhos necessários
            final_offspring = offspring[:num_offspring_to_create]
            
            # 3. Avalia apenas os novos filhos
            self._evaluate_population(Population(final_offspring))
            
            # 4. Nova Geração: A nova população é a elite + os novos filhos
            current_population = Population(elites + final_offspring)

            # 5. Condição de Parada
            if fidelity_threshold:
                best_ind = current_population.get_fittest()
                if best_ind and best_ind.fidelity is not None and best_ind.fidelity >= fidelity_threshold:
                    logging.info(f"  -> Limiar de Fidelidade {fidelity_threshold} atingido na geração {gen}. Finalizando fase.")
                    stopping_reason = "fidelity_threshold_reached"
                    break
        
        final_gen_index = max_generations if stopping_reason == "max_generations_reached" else gen + 1
        phase_duration = time.time() - phase_start_time

        if self._observer:
            self._observer.update(final_gen_index, current_population, self._mutation.mutation_rate, self._crossover.crossover_rate)
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
            key = individual.get_structural_key()
            if key in self._fitness_cache:
                individual.fitness, individual.fidelity = self._fitness_cache[key]
            else:
                individual.fitness, individual.fidelity = self._fitness_evaluator.evaluate(individual)
                self._fitness_cache[key] = (individual.fitness, individual.fidelity)
                self._total_evaluations += 1
        
        if evaluated_count > 0:
            logging.debug(f"Evaluated {evaluated_count} new individuals.")
        
        self._fitness_shaper.shape(population)