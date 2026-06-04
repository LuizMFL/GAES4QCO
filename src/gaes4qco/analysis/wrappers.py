from analysis.profiler import profile_time
from evolutionary_algorithm.interfaces import ISelectionStrategy, IPopulationCrossover, IMutationPopulation
from optimization.interfaces import IFitnessEvaluator, IFitnessShaper


class BaseProfilerWrapper:
    """
    Classe base para wrappers de profiling que propaga o acesso a atributos
    de forma transparente para o objeto decorado.
    """

    def __init__(self, decorated):
        self.__dict__['_decorated'] = decorated

    def __getattr__(self, name):
        return getattr(self._decorated, name)

    def __setattr__(self, name, value):
        setattr(self._decorated, name, value)


class FitnessEvaluatorProfilerWrapper(BaseProfilerWrapper, IFitnessEvaluator):
    @profile_time
    def evaluate(self, circuit):
        return self._decorated.evaluate(circuit)


class FitnessShaperProfilerWrapper(BaseProfilerWrapper, IFitnessShaper):
    @profile_time
    def shape(self, population):
        return self._decorated.shape(population)


class SelectionProfilerWrapper(BaseProfilerWrapper, ISelectionStrategy):
    @profile_time
    def select(self, *args, **kwargs):
        # Repassa todos os argumentos posicionais e nomeados para o método original
        return self._decorated.select(*args, **kwargs)


class CrossoverProfilerWrapper(BaseProfilerWrapper, IPopulationCrossover):
    @profile_time
    def run(self, parent_population):
        return self._decorated.run(parent_population)


class MutationProfilerWrapper(BaseProfilerWrapper, IMutationPopulation):
    @profile_time
    def mutate(self, population):
        return self._decorated.mutate(population)