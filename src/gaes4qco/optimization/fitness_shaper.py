import random

from .interfaces import IFitnessShaper
from evolutionary_algorithm.population import Population
from analysis.interfaces import IDistanceMetric


class NullFitnessShaper(IFitnessShaper):
    """Um modelador que não faz nada."""
    def shape(self, population: Population):
        pass


class FitnessSharingShaper(IFitnessShaper):
    """
    Aplica Fitness Sharing usando amostragem e uma métrica de distância injetada.
    """
    def __init__(self, sharing_radius: float, alpha: float, distance_metric: IDistanceMetric, sample_size):
        self._sigma_share = sharing_radius
        self._alpha = alpha
        self._sample_size = sample_size
        self._distance_metric = distance_metric

    def shape(self, population: Population):
        individuals = population.get_individuals()
        n = len(individuals)
        if n < 2:
            return

        for ind_i in individuals:
            if getattr(ind_i, 'base_fitness', None) is None:
                ind_i.base_fitness = ind_i.fitness

            sample_indices = random.sample(range(n), min(self._sample_size, n))
            
            niche_count = 0
            for j in sample_indices:
                ind_j = individuals[j]
                
                d = self._distance_metric.calculate(ind_i, ind_j)
                
                if d < self._sigma_share:
                    sh = 1 - (d / self._sigma_share) ** self._alpha
                    niche_count += sh

            if niche_count > 1:
                ind_i.fitness = ind_i.base_fitness / niche_count
            else:
                ind_i.fitness = ind_i.base_fitness
