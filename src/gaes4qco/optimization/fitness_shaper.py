from analysis.distance_metrics import LevenshteinCircuitDistance
import random

from .interfaces import IFitnessShaper
from evolutionary_algorithm.population import Population


class NullFitnessShaper(IFitnessShaper):
    """Um modelador que não faz nada."""
    def shape(self, population: Population):
        pass


class FitnessSharingShaper(IFitnessShaper):
    """
    Aplica Fitness Sharing usando amostragem e a distância de Levenshtein.
    """
    def __init__(self, sharing_radius: float, alpha: float, sample_size: int = 50):
        self._sigma_share = sharing_radius
        self._alpha = alpha
        self._sample_size = sample_size
        self._distance_metric = LevenshteinCircuitDistance()

    def shape(self, population: Population):
        individuals = population.get_individuals()
        n = len(individuals)
        if n < 2:
            return

        for ind_i in individuals:
            if ind_i.fitness is None:
                continue

            sample_indices = random.sample(range(n), min(self._sample_size, n))
            
            niche_count = 0
            for j in sample_indices:
                ind_j = individuals[j]
                
                d = self._distance_metric.calculate(ind_i, ind_j)
                
                if d < self._sigma_share:
                    sh = 1 - (d / self._sigma_share) ** self._alpha
                    niche_count += sh
            
            if niche_count > 1:
                ind_i.fitness /= niche_count
