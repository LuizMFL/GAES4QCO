from optimization.interfaces import IRateAdapter
from shared.value_objects import EvolutionRates


class FixedRateAdapter(IRateAdapter):
    """Implementa a estratégia de taxas fixas."""

    def __init__(self, crossover_rate: float, mutation_rate: float):
        self._rates = EvolutionRates(crossover_rate, mutation_rate)

    def adapt(self, diversity: float) -> EvolutionRates:
        # Simplesmente retorna as taxas fixas, ignorando a diversidade.
        return self._rates


class DiversityAdaptiveRateAdapter(IRateAdapter):
    """
    Adapta as taxas de evolução com base na diversidade genética da população.
    """

    def __init__(self, min_mutation_rate: float, max_mutation_rate: float,
                 min_crossover_rate: float, max_crossover_rate: float):
        self.min_mr = min_mutation_rate
        self.max_mr = max_mutation_rate
        self.min_cr = min_crossover_rate
        self.max_cr = max_crossover_rate

    def adapt(self, diversity: float) -> EvolutionRates:
        """
        Calcula as novas taxas usando uma interpolação linear.
        """
        # Normaliza a diversidade para o intervalo [0, 1] (já está nesse intervalo)
        diversity_norm = max(0.0, min(1.0, diversity))

        # Interpolação linear inversa para a mutação
        mutation_rate = self.max_mr - (diversity_norm * (self.max_mr - self.min_mr))

        # Interpolação linear direta para o crossover
        crossover_rate = self.min_cr + (diversity_norm * (self.max_cr - self.min_cr))

        mutation_rate = max(self.min_mr, min(self.max_mr, mutation_rate))
        crossover_rate = max(self.min_cr, min(self.max_cr, crossover_rate))

        return EvolutionRates(crossover_rate=crossover_rate, mutation_rate=mutation_rate)
