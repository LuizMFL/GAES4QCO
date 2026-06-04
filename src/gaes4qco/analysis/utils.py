from dataclasses import is_dataclass, asdict
from typing import Any
from enum import Enum


def dataclass_to_primitive(obj: Any):
    """
    Converte dataclasses aninhadas em dicionários e lida com tipos não-primitivos
    como Enums, garantindo que o resultado seja serializável.
    """
    if is_dataclass(obj):
        # Converte a dataclass para um dicionário
        result = asdict(obj)

        # Itera sobre o dicionário para converter Enums e outros tipos
        for key, value in result.items():
            if isinstance(value, list):
                # Se for uma lista, processa cada item recursivamente
                result[key] = [dataclass_to_primitive(item) for item in value]
            elif isinstance(value, Enum):
                # Se for um Enum, pega seu valor (string)
                result[key] = value.value
            elif is_dataclass(value):
                # Se for outra dataclass aninhada, converte recursivamente
                result[key] = dataclass_to_primitive(value)
        return result

    # Se não for uma dataclass, retorna o objeto como está
    return obj
