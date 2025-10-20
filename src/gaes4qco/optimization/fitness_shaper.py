from analysis.distance_metrics import StructuralJaccardDistance

import numpy as np

from .interfaces import IFitnessShaper
from evolutionary_algorithm.population import Population


class NullFitnessShaper(IFitnessShaper):
    """Um modelador que não faz nada. Usado quando o Fitness Sharing está desativado."""

    def shape(self, population: Population):
        pass  # Não faz nenhuma alteração


class FitnessSharingShaper(IFitnessShaper):
    """
    Aplica Fitness Sharing preservando o melhor indivíduo de cada nicho.
    """

    def __init__(self, sharing_radius: float, alpha: float):
        self._sigma_share = sharing_radius
        self._alpha = alpha
        self._distance_metric = StructuralJaccardDistance()

    def shape(self, population: Population):
        individuals = population.get_individuals()
        n = len(individuals)

        # Passo 1: criar matriz de distâncias (simétrica)
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = self._distance_metric.calculate(individuals[i], individuals[j])
                distances[i, j] = distances[j, i] = d

        # Passo 2: identificar nichos (clusters de indivíduos similares)
        visited = [False] * n
        niches = []

        for i in range(n):
            if visited[i]:
                continue
            # Cria um novo nicho e inclui todos os indivíduos próximos
            niche = [i]
            visited[i] = True
            for j in range(n):
                if not visited[j] and distances[i, j] < self._sigma_share:
                    niche.append(j)
                    visited[j] = True
            niches.append(niche)

        # Passo 3: aplicar penalização dentro de cada nicho
        for niche in niches:
            if len(niche) == 1:
                continue  # nicho unitário → sem penalização

            # Ordenar por fitness (maior primeiro)
            niche_sorted = sorted(niche, key=lambda idx: individuals[idx].fitness, reverse=True)

            best_idx = niche_sorted[0]  # melhor do nicho → preservado

            for idx in niche_sorted[1:]:  # os demais sofrem penalização
                niche_count = 0
                for jdx in niche:
                    d = distances[idx, jdx]
                    if d < self._sigma_share:
                        sh = 1 - (d / self._sigma_share) ** self._alpha
                        niche_count += sh

                if niche_count > 0:
                    individuals[idx].fitness /= niche_count
