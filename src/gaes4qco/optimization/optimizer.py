from typing import List, Optional

from evolutionary_algorithm.population_factory import PopulationFactory
from evolutionary_algorithm.interfaces import ISelectionStrategy, IMutationPopulation, IPopulationCrossover
from evolutionary_algorithm.population import Population
from evolutionary_algorithm.rate_adapter import IRateAdapter
from .interfaces import IFitnessEvaluator, IProgressObserver, IFitnessShaper


class Optimizer:
    """
    ## O motor do Algoritmo Genético. Orquestra todo o processo evolucionário.
    ## É configurado com interfaces, não com classes concretas.
    """

    def __init__(
            self,
            fitness_evaluator: IFitnessEvaluator,
            parent_selection: ISelectionStrategy,
            survivor_selection: ISelectionStrategy,
            crossover: IPopulationCrossover,
            mutation: IMutationPopulation,
            population_factory: PopulationFactory,
            rate_adapter: IRateAdapter,
            diversity_threshold: float,
            injection_rate: float,
            fitness_shaper: IFitnessShaper,
            observer: IProgressObserver
    ):
        self._fitness_evaluator = fitness_evaluator
        self._parent_selection = parent_selection
        self._survivor_selection = survivor_selection
        self._crossover = crossover
        self._mutation = mutation
        self._population_factory = population_factory
        self._rate_adapter = rate_adapter
        self._diversity_threshold = diversity_threshold
        self._injection_rate = injection_rate
        self._fitness_shaper = fitness_shaper
        self._observer = observer

    def run(
            self,
            initial_population: Population,
            max_generations: int,
            fidelity_threshold: Optional[float]
    ) -> Population:
        """
        Executa o fluxo do algoritmo genético por um número de gerações.
        """
        current_population = initial_population

        print("Evaluating initial population...")
        self._evaluate_population(current_population)
        for gen in range(max_generations):
            if self._observer:
                self._observer.update(gen, current_population)
            current_diversity = current_population.calculate_structural_diversity()
            current_rates = self._rate_adapter.adapt(current_diversity)
            self._crossover.crossover_rate = current_rates.crossover_rate
            self._mutation.mutation_rate = current_rates.mutation_rate
            # 1. Seleção dos Pais
            parent_population = self._parent_selection.select(current_population)

            # 2. Cruzamento
            offspring_population = self._crossover.run(parent_population)

            # 3. Mistura a antiga população com a nova, evitando duplicatas
            offspring_population = Population(offspring_population.get_individuals() + current_population.get_individuals())
            population_without_duplicates = offspring_population.without_duplicates()

            # 4. Mutação
            mutated_population = self._mutation.mutate(population_without_duplicates)

            # 5. Avaliação dos novos indivíduos
            self._evaluate_population(mutated_population)

            current_population = self._survivor_selection.select(mutated_population)

            if fidelity_threshold:
                best_ind = current_population.get_fittest()
                if best_ind and best_ind.fidelity >= fidelity_threshold:
                    print(f"  -> Limiar de Fidelidade {fidelity_threshold} atingido na geração {gen}. Finalizando fase.")
                    break

        if self._observer:
            self._observer.update(max_generations, current_population)
            self._observer.save()

        return current_population

    def _evaluate_population(self, population: Population):
        """
        ## Helper para calcular o fitness de cada indivíduo que ainda não foi avaliado.
        ## Substitui a antiga função 'applyFitnessIntoCircuit'.
        """
        for individual in population.get_individuals():
            individual.fitness, individual.fidelity = self._fitness_evaluator.evaluate(individual)
        self._fitness_shaper.shape(population)
