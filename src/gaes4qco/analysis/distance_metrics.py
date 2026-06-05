from .interfaces import IDistanceMetric
from quantum_circuit.circuit import Circuit


def levenshtein_distance(seq1, seq2):
    """
    Calcula a distância de Levenshtein entre duas sequências.
    O custo de inserção, exclusão e substituição é 1.
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = [[0] * size_y for _ in range(size_x)]
    for x in range(size_x):
        matrix[x][0] = x
    for y in range(size_y):
        matrix[0][y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x][y] = matrix[x-1][y-1]
            else:
                matrix[x][y] = min(
                    matrix[x-1][y] + 1,      # Deleção
                    matrix[x-1][y-1] + 1,  # Substituição
                    matrix[x][y-1] + 1,      # Inserção
                )
    return matrix[size_x - 1][size_y - 1]


class LevenshteinCircuitDistance(IDistanceMetric):
    """
    Calcula a distância topológica entre dois circuitos usando a distância de Levenshtein
    em suas sequências de colunas (chaves estruturais), respeitando a ordem.
    """
    @staticmethod
    def calculate(circuit1: Circuit, circuit2: Circuit) -> float:
        # Obtém as chaves estruturais, que são sequências ordenadas de colunas
        key1 = circuit1.get_structural_key()
        key2 = circuit2.get_structural_key()
        
        # Calcula a distância de edição entre as duas sequências
        distance = levenshtein_distance(key1, key2)
        
        # Normaliza a distância para o intervalo [0, 1] para ser uma métrica consistente
        max_len = max(len(key1), len(key2))
        if max_len == 0:
            return 0.0
        return distance / max_len
