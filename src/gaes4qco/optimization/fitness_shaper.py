from analysis.distance_metrics import StructuralJaccardDistance
import numpy as np
import random

from .interfaces import IFitnessShaper
from evolutionary_algorithm.population import Population


class NullFitnessShaper(IFitnessShaper):
    """Um modelador que não faz nada. Usado quando o Fitness Sharing está desativado."""
    def shape(self, population: Population):
        pass


class FitnessSharingShaper(IFitnessShaper):
    """
    Aplica Fitness Sharing usando amostragem para otimizar a performance.
    """
    def __init__(self, sharing_radius: float, alpha: float, sample_size: int = 50):
        self._sigma_share = sharing_radius
        self._alpha = alpha
        self._sample_size = sample_size
        self._distance_metric = StructuralJaccardDistance()

    def shape(self, population: Population):
        individuals = population.get_individuals()
        if len(individuals) < 2:
            return

        for ind_i in individuals:
            # Garante que o fitness não seja None para evitar erros
            if ind_i.fitness is None:
                continue

            sample_indices = random.sample(range(len(individuals)), min(self._sample_size, len(individuals)))
            
            niche_count = 0
            for j in sample_indices:
                ind_j = individuals[j]
                
                d = self._distance_metric.calculate(ind_i, ind_j)
                
                if d < self._sigma_share:
                    sh = 1 - (d / self._sigma_share) ** self._alpha
                    niche_count += sh
            
            # O nicho de um indivíduo sempre inclui ele mesmo, então o count é no mínimo 1.
            # Isso previne a divisão por valores muito pequenos, estabilizando o fitness.
            if niche_count > 1:
                ind_i.fitness /= niche_count