from .interfaces import IDistanceMetric
from quantum_circuit.circuit import Circuit


class StructuralJaccardDistance(IDistanceMetric):
    """
    Calcula a distância estrutural usando a métrica de Jaccard.
    Ignora parâmetros de gate e usa apenas a estrutura.
    """
    @staticmethod
    def calculate(ind1: Circuit, ind2: Circuit) -> float:
        set1 = ind1.get_structural_representation()
        set2 = ind2.get_structural_representation()

        intersection_size = len(set1.intersection(set2))
        union_size = len(set1.union(set2))

        if union_size == 0:
            return 0.0

        jaccard_similarity = intersection_size / union_size
        return 1.0 - jaccard_similarity
