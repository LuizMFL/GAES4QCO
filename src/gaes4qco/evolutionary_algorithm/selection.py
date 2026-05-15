import random
from abc import ABC, abstractmethod
from typing import List
from enum import Enum

from quantum_circuit.circuit import Circuit
from .interfaces import ISelectionStrategy
from .population import Population


class SelectionType(Enum):
    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    NSGA2 = "nsga2"


class TournamentParentSelection(ISelectionStrategy):
    def __init__(self, population_size: int, tournament_size: int):
        self.population_size = population_size
        self.tournament_size = tournament_size

    def select(self, population: Population) -> Population:
        if not population:
            return Population()

        individuals = population.get_individuals()
        next_gen_parents = []
        for _ in range(self.population_size):
            group = random.sample(individuals, min(self.tournament_size, len(individuals)))
            champion = max(group, key=lambda ind: ind.fitness)
            next_gen_parents.append(champion)
        return Population(next_gen_parents)


class TournamentSurvivorSelection(ISelectionStrategy):
    def __init__(self, population_size: int, tournament_size: int, elitism_count: int):
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count

    def select(self, population: Population) -> Population:
        if not population:
            return Population()

        individuals = population.get_individuals()
        sorted_population = sorted(individuals, key=lambda ind: ind.fidelity, reverse=True)

        elites = sorted_population[:self.elitism_count]
        competitors = sorted_population[self.elitism_count:]
        survivors = elites[:]

        while len(survivors) < self.population_size and competitors:
            group = random.sample(competitors, min(self.tournament_size, len(competitors)))
            champion = max(group, key=lambda ind: ind.fitness)
            survivors.append(champion)
            competitors.remove(champion)

        return Population(survivors)


class RandomParentSelection(ISelectionStrategy):
    def __init__(self, population_size: int):
        self.population_size = population_size

    def select(self, population: Population) -> Population:
        if not population:
            return Population()
        individuals = population.get_individuals()
        selected = random.choices(individuals, k=self.population_size)
        return Population(selected)


class RandomSurvivorSelection(ISelectionStrategy):
    def __init__(self, population_size: int, elitism_count: int = 1):
        self.population_size = population_size
        self.elitism_count = elitism_count

    def select(self, population: Population) -> Population:
        if not population:
            return Population()

        individuals = population.get_individuals()
        elites = sorted(individuals, key=lambda ind: ind.fidelity, reverse=True)[:self.elitism_count]
        remaining_slots = self.population_size - len(elites)
        survivors = elites + random.sample(individuals, min(remaining_slots, len(individuals)))
        return Population(survivors)


class RouletteParentSelection(ISelectionStrategy):
    def __init__(self, population_size: int):
        self.population_size = population_size

    def select(self, population: Population) -> Population:
        if not population:
            return Population()

        individuals = population.get_individuals()
        total_fitness = sum(ind.fitness for ind in individuals)
        if total_fitness == 0:
            # fallback: random uniforme
            selected = random.choices(individuals, k=self.population_size)
        else:
            weights = [ind.fitness / total_fitness for ind in individuals]
            selected = random.choices(individuals, weights=weights, k=self.population_size)
        return Population(selected)


class RouletteSurvivorSelection(ISelectionStrategy):
    def __init__(self, population_size: int, elitism_count: int = 1):
        self.population_size = population_size
        self.elitism_count = elitism_count

    def select(self, population: Population) -> Population:
        if not population:
            return Population()

        individuals = population.get_individuals()
        elites = sorted(individuals, key=lambda ind: ind.fidelity, reverse=True)[:self.elitism_count]

        competitors = [ind for ind in individuals if ind not in elites]
        total_fitness = sum(ind.fitness for ind in competitors)

        survivors = list(elites)

        if total_fitness == 0:
            survivors.extend(random.choices(competitors, k=self.population_size - len(survivors)))
        else:
            weights = [ind.fitness / total_fitness for ind in competitors]
            survivors.extend(random.choices(competitors, weights=weights, k=self.population_size - len(survivors)))

        return Population(survivors)


class IMultiObjectiveService(ABC):
    """
    Interface for multi-objective optimization services.
    Defines the contract for non-dominated sorting, Pareto dominance,
    and crowding distance assignment.
    """

    @abstractmethod
    def non_dominated_sort(self, individuals: List[Circuit]) -> List[List[Circuit]]:
        """Perform non-dominated sorting and return Pareto fronts."""
        pass

    @abstractmethod
    def dominates(self, p: Circuit, q: Circuit) -> bool:
        """Return True if p Pareto-dominates q."""
        pass

    @abstractmethod
    def crowding_distance_assignment(self, front: List[Circuit]) -> None:
        """Assign crowding distances to individuals in a given front."""
        pass


class NSGA2Service(IMultiObjectiveService):
    """Core NSGA-II operators (non-dominated sort, dominance check, crowding distance)."""

    def non_dominated_sort(self, individuals: List[Circuit]) -> List[List[Circuit]]:
        fronts = [[]]
        for p in individuals:
            p.domination_count = 0
            p.dominated_solutions = []
            for q in individuals:
                if p is q:
                    continue
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
        if not front:
            return
        for ind in front:
            ind.crowding_distance = 0.0
        num_objectives = len(front[0].objectives)
        for m in range(num_objectives):
            front.sort(key=lambda x: x.objectives[m])
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            min_obj, max_obj = front[0].objectives[m], front[-1].objectives[m]
            range_obj = max_obj - min_obj
            if range_obj == 0:
                continue
            for i in range(1, len(front) - 1):
                distance = front[i + 1].objectives[m] - front[i - 1].objectives[m]
                front[i].crowding_distance += distance / range_obj


class NSGA2SurvivorSelection(ISelectionStrategy):
    def __init__(self, population_size: int, elitism_count: int, nsga2_service: IMultiObjectiveService):
        self.population_size = population_size
        self.elitism_count = max(elitism_count, 1)
        self._nsga2 = nsga2_service

    def select(self, population: Population) -> Population:
        individuals = population.get_individuals()
        if not individuals:
            return Population()

        sorted_by_fidelity = sorted(individuals, key=lambda ind: ind.fidelity, reverse=True)
        elites = sorted_by_fidelity[:self.elitism_count]

        remaining = sorted_by_fidelity[self.elitism_count:]
        fronts = self._nsga2.non_dominated_sort(remaining)

        survivors = list(elites)
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
