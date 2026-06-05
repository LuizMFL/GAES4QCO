import random
from abc import ABC, abstractmethod
from typing import List, Optional
from enum import Enum

from quantum_circuit.circuit import Circuit
from .interfaces import ISelectionStrategy
from .population import Population


class SelectionType(Enum):
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    NSGA2 = "nsga2"
    RANDOM = "random"


def _get_evaluated_individuals(population: Population) -> List[Circuit]:
    """Helper para retornar apenas indivíduos que já foram avaliados."""
    return [ind for ind in population.get_individuals() if ind.fitness is not None]


class TournamentParentSelection(ISelectionStrategy):
    def __init__(self, tournament_size: int):
        self.tournament_size = tournament_size

    def select(self, population: Population, num_to_select: Optional[int] = None) -> Population:
        individuals = _get_evaluated_individuals(population)
        if not individuals:
            return Population()
        
        # Se num_to_select não for fornecido, assume o tamanho da população atual
        k = num_to_select if num_to_select is not None else len(individuals)

        next_gen_parents = []
        for _ in range(k):
            group = random.sample(individuals, min(self.tournament_size, len(individuals)))
            champion = max(group, key=lambda ind: ind.fitness)
            next_gen_parents.append(champion)
        return Population(next_gen_parents)


class TournamentSurvivorSelection(ISelectionStrategy):
    def __init__(self, population_size: int, tournament_size: int, elitism_count: int):
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count

    def select(self, population: Population, num_to_select: Optional[int] = None) -> Population:
        individuals = _get_evaluated_individuals(population)
        if not individuals:
            return Population()

        individuals.sort(key=lambda ind: ind.fitness, reverse=True)
        elites = individuals[:self.elitism_count]
        competitors = individuals[self.elitism_count:]
        survivors = elites[:]
        
        while len(survivors) < self.population_size and competitors:
            group = random.sample(competitors, min(self.tournament_size, len(competitors)))
            champion = max(group, key=lambda ind: ind.fitness)
            survivors.append(champion)
            competitors.remove(champion)

        return Population(survivors)


class RandomParentSelection(ISelectionStrategy):
    def select(self, population: Population, num_to_select: Optional[int] = None) -> Population:
        individuals = _get_evaluated_individuals(population)
        if not individuals:
            return Population()
        
        k = num_to_select if num_to_select is not None else len(individuals)
        selected = random.choices(individuals, k=k)
        return Population(selected)


class RandomSurvivorSelection(ISelectionStrategy):
    def __init__(self, population_size: int, elitism_count: int = 1):
        self.population_size = population_size
        self.elitism_count = elitism_count

    def select(self, population: Population, num_to_select: Optional[int] = None) -> Population:
        individuals = _get_evaluated_individuals(population)
        if not individuals:
            return Population()

        individuals.sort(key=lambda ind: ind.fitness, reverse=True)
        elites = individuals[:self.elitism_count]
        
        competitors = [ind for ind in individuals if ind not in elites]
        num_to_select_randomly = self.population_size - len(elites)
        
        if num_to_select_randomly <= 0:
            return Population(elites)
            
        random_survivors = random.sample(competitors, min(num_to_select_randomly, len(competitors)))
        
        return Population(elites + random_survivors)


class RouletteParentSelection(ISelectionStrategy):
    def select(self, population: Population, num_to_select: Optional[int] = None) -> Population:
        individuals = _get_evaluated_individuals(population)
        if not individuals:
            return Population()

        k = num_to_select if num_to_select is not None else len(individuals)
        
        total_fitness = sum(ind.fitness for ind in individuals)
        if total_fitness == 0:
            selected = random.choices(individuals, k=k)
        else:
            weights = [ind.fitness / total_fitness for ind in individuals]
            selected = random.choices(individuals, weights=weights, k=k)
        return Population(selected)


class RouletteSurvivorSelection(ISelectionStrategy):
    def __init__(self, population_size: int, elitism_count: int = 1):
        self.population_size = population_size
        self.elitism_count = elitism_count

    def select(self, population: Population, num_to_select: Optional[int] = None) -> Population:
        individuals = _get_evaluated_individuals(population)
        if not individuals:
            return Population()

        individuals.sort(key=lambda ind: ind.fitness, reverse=True)
        elites = individuals[:self.elitism_count]
        
        competitors = [ind for ind in individuals if ind not in elites]
        num_to_select_roulette = self.population_size - len(elites)
        if num_to_select_roulette <= 0:
            return Population(elites)

        total_fitness = sum(ind.fitness for ind in competitors)
        if total_fitness == 0 or not competitors:
            selected = random.choices(competitors, k=num_to_select_roulette) if competitors else []
        else:
            weights = [ind.fitness / total_fitness for ind in competitors]
            selected = random.choices(competitors, weights=weights, k=num_to_select_roulette)
            
        return Population(elites + selected)


class IMultiObjectiveService(ABC):
    @abstractmethod
    def non_dominated_sort(self, individuals: List[Circuit]) -> List[List[Circuit]]:
        pass

    @abstractmethod
    def dominates(self, p: Circuit, q: Circuit) -> bool:
        pass

    @abstractmethod
    def crowding_distance_assignment(self, front: List[Circuit]) -> None:
        pass


class NSGA2Service(IMultiObjectiveService):
    def non_dominated_sort(self, individuals: List[Circuit]) -> List[List[Circuit]]:
        fronts = [[]]
        for p in individuals:
            p.domination_count = 0
            p.dominated_solutions = []
            for q in individuals:
                if p is q: continue
                if self.dominates(p, q):
                    p.dominated_solutions.append(q)
                elif self.dominates(q, p):
                    p.domination_count += 1
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)

        i = 0
        while i < len(fronts) and fronts[i]:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)
        return fronts

    def dominates(self, p: Circuit, q: Circuit) -> bool:
        p_objectives, q_objectives = p.objectives, q.objectives
        at_least_one_better = any(p_obj > q_obj for p_obj, q_obj in zip(p_objectives, q_objectives))
        none_worse = all(p_obj >= q_obj for p_obj, q_obj in zip(p_objectives, q_objectives))
        return at_least_one_better and none_worse

    def crowding_distance_assignment(self, front: List[Circuit]):
        if not front: return
        for ind in front:
            ind.crowding_distance = 0.0
        num_objectives = len(front[0].objectives)
        for m in range(num_objectives):
            front.sort(key=lambda x: x.objectives[m])
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            min_obj, max_obj = front[0].objectives[m], front[-1].objectives[m]
            range_obj = max_obj - min_obj
            if range_obj == 0: continue
            for i in range(1, len(front) - 1):
                distance = front[i + 1].objectives[m] - front[i - 1].objectives[m]
                front[i].crowding_distance += distance / range_obj


class NSGA2SurvivorSelection(ISelectionStrategy):
    def __init__(self, population_size: int, nsga2_service: IMultiObjectiveService, elitism_count: int = 0):
        self.population_size = population_size
        self._nsga2 = nsga2_service

    def select(self, population: Population, num_to_select: Optional[int] = None) -> Population:
        individuals = _get_evaluated_individuals(population)
        if not individuals:
            return Population()

        fronts = self._nsga2.non_dominated_sort(individuals)

        survivors = []
        for front in fronts:
            if len(survivors) + len(front) <= self.population_size:
                survivors.extend(front)
            else:
                self._nsga2.crowding_distance_assignment(front)
                front.sort(key=lambda x: x.crowding_distance, reverse=True)
                needed = self.population_size - len(survivors)
                survivors.extend(front[:needed])
                break
        
        return Population(survivors)
