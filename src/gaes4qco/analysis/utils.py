from dataclasses import is_dataclass, asdict
from typing import Any


def dataclass_to_primitive(obj: Any):
    """
    Converte dataclasses aninhadas em dicionários práticos para uso por plotters / JSON.
    Usa dataclasses.asdict internamente, mas garante que enums e objetos não-serializáveis
    sejam transformados apropriadamente se necessário.
    """
    if is_dataclass(obj):
        return asdict(obj)
    return obj
