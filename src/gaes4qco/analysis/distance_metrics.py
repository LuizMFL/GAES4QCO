from .interfaces import IDistanceMetric
from quantum_circuit.circuit import Circuit


class StructuralJaccardDistance(IDistanceMetric):
    """
    Calcula a distância estrutural usando a métrica de Jaccard.
    A comparação é feita tratando cada coluna (com seus gates ordenados) como um elemento único.
    """
    @staticmethod
    def calculate(ind1: Circuit, ind2: Circuit) -> float:
        # Usa o novo método com cache para obter os conjuntos diretamente
        set_of_columns1 = ind1.get_structural_set()
        set_of_columns2 = ind2.get_structural_set()

        intersection_size = len(set_of_columns1.intersection(set_of_columns2))
        union_size = len(set_of_columns1.union(set_of_columns2))

        if union_size == 0:
            return 0.0

        jaccard_similarity = intersection_size / union_size
        return 1.0 - jaccard_similarity