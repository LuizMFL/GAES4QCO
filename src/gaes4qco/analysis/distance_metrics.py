from .interfaces import IDistanceMetric

def levenshtein_distance(seq1, seq2):
    """
    Calcula a distância de Levenshtein entre duas sequências.
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
                    matrix[x-1][y] + 1,
                    matrix[x-1][y-1] + 1,
                    matrix[x][y-1] + 1,
                )
    return matrix[size_x - 1][size_y - 1]


class LevenshteinCircuitDistance(IDistanceMetric):
    """
    Calcula a distância topológica entre dois circuitos usando a distância de Levenshtein.
    """
    @staticmethod
    def calculate(circuit1, circuit2, **kwargs) -> float:
        # Obtém as chaves estruturais diretamente dos circuitos
        key1 = circuit1.get_structural_key()
        key2 = circuit2.get_structural_key()
        
        distance = levenshtein_distance(key1, key2)
        
        max_len = max(len(key1), len(key2))
        if max_len == 0:
            return 0.0
        return distance / max_len