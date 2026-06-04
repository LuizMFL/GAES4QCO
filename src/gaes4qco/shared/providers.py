from dependency_injector import providers
from analysis.profiler import profile_time

class ProfilingDecoratorProvider(providers.Provider):
    """
    Um Provider customizado que instancia outro provider e depois aplica
    o decorator @profile_time a todos os métodos públicos do objeto instanciado.
    """
    def __init__(self, provider=None):
        self._provider = provider
        super().__init__()

    def _provide(self, *args, **kwargs):
        # 1. Instancia o objeto usando o provider original
        instance = self._provider(*args, **kwargs)

        # 2. Itera sobre os atributos do objeto instanciado
        for name in dir(instance):
            # Ignora métodos privados/mágicos
            if not name.startswith('_'):
                attr = getattr(instance, name)
                if callable(attr) and not isinstance(attr, type):
                    # Aplica o decorator e substitui o método original
                    setattr(instance, name, profile_time(attr))
        
        return instance

    def __deepcopy__(self, memo):
        # Sobrescreve o deepcopy para lidar com a cópia do provider aninhado
        copied = super().__deepcopy__(memo)
        copied._provider = providers.deepcopy(self._provider, memo)
        return copied