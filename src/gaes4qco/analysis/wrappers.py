from analysis.profiler import profile_time
from optimization.interfaces import IFitnessEvaluator, IFitnessShaper, IRateAdapter
from evolutionary_algorithm.interfaces import ISelectionStrategy, IPopulationCrossover, IMutationPopulation
from analysis.interfaces import IDistanceMetric


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
    def shape(self, *args, **kwargs):
        # Repassa todos os argumentos posicionais e nomeados
        return self._decorated.shape(*args, **kwargs)


class SelectionProfilerWrapper(BaseProfilerWrapper, ISelectionStrategy):
    @profile_time
    def select(self, *args, **kwargs):
        return self._decorated.select(*args, **kwargs)


class CrossoverProfilerWrapper(BaseProfilerWrapper, IPopulationCrossover):
    @profile_time
    def run(self, parent_population):
        return self._decorated.run(parent_population)


class MutationProfilerWrapper(BaseProfilerWrapper, IMutationPopulation):
    @profile_time
    def mutate(self, population):
        return self._decorated.mutate(population)


class RateAdapterProfilerWrapper(BaseProfilerWrapper, IRateAdapter):
    @profile_time
    def adapt(self, diversity: float):
        return self._decorated.adapt(diversity)


class DistanceMetricProfilerWrapper(BaseProfilerWrapper, IDistanceMetric):
    @profile_time
    def calculate(self, *args, **kwargs):
        return self._decorated.calculate(*args, **kwargs)